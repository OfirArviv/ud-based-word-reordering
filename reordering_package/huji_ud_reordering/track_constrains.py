import pathlib
from copy import deepcopy
# from random import shuffle

import UDLib as U
import Estimation as E


def compare_orderings(dict1, dict2, tree_no):
    '''
    Checks if new constraints were added to the
    dict or if the polarities of some of the old constraints
    were reversed. dict2 is guaranteed to have all the
    constraints from dict1.
    '''
    new_deprels = set(dict2)-set(dict1)
    for deprel in new_deprels:
        print(f"{tree_no:5}: new expansion: {deprel}")
    # Don't look for minor differences
    if new_deprels:
        return

    for deprel in dict2:
        for key in dict2[deprel]:
            if key not in dict1[deprel]:
                print(f"{tree_no:5}: new constraint: {deprel} -> {key}, {dict2[deprel][key]}")
            elif dict1[deprel][key]*dict2[deprel][key] < 0:
                print(f"{tree_no:5}: constraint polarity reversed: {deprel} -> {key}, {dict2[deprel][key]}")


def conllu2trees(path):
    with open(path, 'r') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [U.UDTree(*U.conll2graph(block)) for block in blocks]


def track_constraints(path, burn_in):
    trees = conllu2trees(path)
    print(f"{len(trees)} trees")
    # idx_ja = [i for i in range(len(trees))]

    # First run: no shuffling
    estimates = {}
    for i in range(burn_in):
        root_idx = trees[i].get_real_root()  # '0' is not in the node index
        E.process_subtree(
            trees[i],
            root_idx,
            estimates,
            True
        )
    print(f"Encountered {len(estimates)} expansions during burn in: {', '.join(sorted(estimates))}")
    # After burn in, compare new estimates on each iteration
    for i in range(burn_in, len(trees)):
        estimates_old = deepcopy(estimates)
        root_idx = trees[i].get_real_root()
        E.process_subtree(
            trees[i],
            root_idx,
            estimates,
            True
        )
        compare_orderings(estimates_old, estimates, i+1)


data_dir = pathlib.Path('data')

print('German')
track_constraints(data_dir / 'de_gsd-ud-train.conllu', 1000)
# track_constraints(data_dir / 'ga_idt-ud-dev.conllu', 500)
