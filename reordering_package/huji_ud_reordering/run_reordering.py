import argparse
import json
import time
from tqdm import tqdm
import UDLib as U
import ReorderingNew as R
from p_tqdm import p_map
import multiprocessing as mp


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-i", "--input", required=True)
    argparser.add_argument("-e", "--estimates", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)
    argparser.add_argument("--no-with-gurobi", action='store_true', default=False)
    argparser.add_argument("--no-separate-neg", action='store_true', default=False)
    argparser.add_argument("--preference-threshold", type=float,  default=0.8)
    argparser.add_argument('--disable-parallel', action='store_true', default=False)

    args = argparser.parse_args()
    filename = args.input.split("/")[-1]
    estimates_name = args.estimates.split("/")[-1]
    output_path = f'{args.output_dir}/{filename}' \
                  f'.reordered_by.{estimates_name}' \
                  f'{".no_with_gurobi" if args.no_with_gurobi else ""}' \
                  f'{".no_separate_neg" if args.no_separate_neg else ""}' \
                  f'{"" if args.no_with_gurobi else ".preference_threshold-"+str(args.preference_threshold)}' \
                  f'.conllu'

    input_trees = U.conllu2trees(args.input)
    with open(args.estimates, 'r', encoding='utf-8') as inp:
        estimates = json.load(inp)

    start_time = time.time()
    reordered_trees = []
    if args.disable_parallel:
        print("Running sequentially...")
        for i, t in enumerate(tqdm(input_trees)):
            try:
                reordered_trees.append(R.reorder_tree(t, estimates))
            except Exception as e:
                print(f'Reordering failed in tree #{i}: {e}')
    else:
        print("Running in parallel...")
        print(f'Num of available processors: {mp.cpu_count()}')

        def reorder_tree_wrapper(input):
            # tree: U.UDTree, i: int, estimates: Dict[str, Dict[Tuple[str, str], int]]
            (tree, i, estimates) = input
            try:
                return R.reorder_tree(tree, estimates, separate_neg=not args.no_separate_neg,
                                      with_gurobi=not args.no_with_gurobi,
                                      preference_threshold=args.preference_threshold)
            except Exception as e:
                print(f'Reordering failed on tree #{i}: {e}')
                return


        reordered_trees = p_map(reorder_tree_wrapper, [(t, i, estimates) for i, t in enumerate(input_trees)])
        reordered_trees = filter(lambda x: x is not None, reordered_trees)

    with open(output_path, 'w', encoding='utf-8') as out:
        print('\n\n'.join(str(t) for t in reordered_trees), file=out)

    print(f'Done. Duration:{(time.time() - start_time)}')
