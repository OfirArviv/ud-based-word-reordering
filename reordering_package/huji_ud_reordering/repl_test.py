import json
import importlib
import pathlib
from collections import Counter

import UDLib as U
import Estimation as E
import Reordering as R

import ilp

def refresh_imports():
    importlib.reload(U)
    importlib.reload(E)
    importlib.reload(R)
    importlib.reload(ilp)


def reorder_treebank(treebank, estimates):
    reordered = []
    error_log = Counter()
    for i, tree in enumerate(treebank):
        try:
            reordered.append(R.reorder(tree, estimates, True, error_log))
        except ValueError as e:
            print(i, e)
            continue
    return reordered, error_log


def reorder_ewt(estimates, lang_code):
    error_log = Counter()
    for i, treebank in enumerate([
        trees_en_ewt_train,
        trees_en_ewt_dev,
        trees_en_ewt_test
    ]):
        reordered = []
        for j, tree in enumerate(treebank):
            try:
                reordered.append(R.reorder(tree, estimates, True, error_log))
            except ValueError as e:
                print(j, e)
                continue
        with open(ouput_dir / f'en_{["ewt_train", "ewt_dev", "ewt_test"][i]}_reordered_{lang_code}_new.conllu', 'w') as out:
            out.write('\n\n'.join(str(t) for t in reordered))
    return error_log


def conllu2trees(path):
    with open(path, 'r', encoding='utf-8') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [U.UDTree(*U.conll2graph(block)) for block in blocks]


def report_reordering_progress(treebank, estimates):
    importlib.reload(R)
    for i, t in enumerate(treebank):
        print(i)
        try:
            _ = R.reorder(t, estimates, True)
        except ValueError as e:
            print(e)
            continue


data_dir = pathlib.Path('data')
ouput_dir = pathlib.Path(data_dir / 'reordered')
estimates_dir = pathlib.Path(data_dir / 'estimates')

# English dataset
trees_en_gum_train = conllu2trees(data_dir / 'en_gum-ud-train.conllu')
trees_en_gum_dev = conllu2trees(data_dir / 'en_gum-ud-dev.conllu')
trees_en_gum_test = conllu2trees(data_dir / 'en_gum-ud-test.conllu')

trees_en_ewt_train = conllu2trees(data_dir / 'en_ewt-ud-train.conllu')
trees_en_ewt_dev = conllu2trees(data_dir / 'en_ewt-ud-dev.conllu')
trees_en_ewt_test = conllu2trees(data_dir / 'en_ewt-ud-test.conllu')

trees_en_pud = conllu2trees(data_dir / 'en_pud-ud-test.conllu')

for i, t in enumerate(trees_en_gum_test):
    if 'not' in str(t):
        print(i, t.get_sentence())
        input()


def dump_estimates(est_dict, path):
    est_dict_str_keys = {}
    for deprel in est_dict:
        est_dict_str_keys[deprel] = {
            f'{r1}->{r2}': count for (r1, r2), count in est_dict[deprel].items()
        }
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(est_dict_str_keys, out, indent=2)

# Learn Japanese ordering and apply it to English.
trees_ja = conllu2trees(data_dir / 'ja_pud-ud-test.conllu')
estimates_ja = E.get_ml_directionality_estimates(trees_ja)
dump_estimates(estimates_ja, estimates_dir / 'japanese_pud.json')

result = ilp.ordering2indices(
    ilp.get_pairwise_ordering(estimates_ja['root'])
)
for deprel, idx in sorted(result.items(), key = lambda x: x[1]):
    print(f'{deprel}: {idx}')


# Learn German ordering and apply it to English.
trees_de = conllu2trees(data_dir / 'de_gsd-ud-train.conllu')
estimates_de = E.get_ml_directionality_estimates(trees_de)
dump_estimates(estimates_de, estimates_dir / 'german_gsd.json')


ewt_en_de_test, ewt_de_test_error = reorder_treebank(trees_en_ewt_test, estimates_de)

error_log_de = reorder_ewt(estimates_de, 'de')

# Learn Arabic ordering and apply it to English.
trees_ar = conllu2trees(data_dir / 'ar_pud-ud-test.conllu')
estimates_ar = E.get_ml_directionality_estimates(trees_ar)
dump_estimates(estimates_ar, estimates_dir / 'arabic_pud.json')

result = ilp.ordering2indices(
    ilp.get_pairwise_ordering(estimates_ar['root'])
)
for deprel, idx in sorted(result.items(), key = lambda x: x[1]):
    print(f'{deprel}: {idx}')



error_log_ar = reorder_ewt(estimates_ar, 'ar')

# Learn Irish ordering and apply it to English.
trees_ga = conllu2trees(data_dir / 'ga_idt_all.conllu')
estimates_ga = E.get_ml_directionality_estimates(trees_ga)
dump_estimates(estimates_ga, estimates_dir / 'irish_idt.json')


result = ilp.ordering2indices(
    ilp.get_pairwise_ordering(estimates_ga['root'])
)
for deprel, idx in sorted(result.items(), key = lambda x: x[1]):
    print(f'{deprel}: {idx}')


error_log_ga = reorder_ewt(estimates_ga, 'ga')

ewt_test_ga, error_log_ga_test = reorder_treebank(trees_en_ewt_test, estimates_ga)

for i in range(20):
    print(ewt_en_de_test[i].get_sentence())

importlib.reload(U)
for i in range(30, 41):
    trees_en_ewt_test[i].get_sentence()
    print("Irish:", R.reorder(trees_en_ewt_test[i], estimates_ga).get_sentence())
    print("Arabic:",R.reorder(trees_en_ewt_test[i], estimates_ar).get_sentence())
    print("Japanese:",R.reorder(trees_en_ewt_test[i], estimates_ja).get_sentence())
    print()


