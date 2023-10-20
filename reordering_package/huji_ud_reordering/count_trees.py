import json

from multiprocessing import Pool, cpu_count

import UDLib as U
import ReorderingNew as R

trees = U.conllu2trees('data/en_ewt-ud-test.conllu')
with open('data/estimates/arabic_pud.json', 'r', encoding='utf-8') as inp:
    estimates_ja = json.load(inp)

tree_count = 0


def tree_was_reordered(tree):
    global tree_count
    tree_count += 1
    print(f"{tree_count}".rjust(5), end='\r')
    return str(tree) == str(R.reorder_tree(tree, estimates_ja, True, True, 0.6))


same = different = 0
result = []
n_cpus = cpu_count()
with Pool(n_cpus) as p:
    for i, res in enumerate(p.map(
        tree_was_reordered,
        trees,
        len(trees) // n_cpus
    )):
        result.append(res)

print()

for r in result:
    if r:
        same += 1
    else:
        different += 1

print(same, different, same/2077*100)
