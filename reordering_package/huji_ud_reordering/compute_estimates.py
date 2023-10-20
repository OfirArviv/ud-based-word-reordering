# Computes the estimates for corpora and dumps them into JSON files

import json
import random
from typing import Optional

import UDLib as U
import Estimation as E
import argparse


def conllu2trees(path):
    with open(path, 'r', encoding='utf-8') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [U.UDTree(*U.conll2graph(block)) for block in blocks]


def dump_estimates(est_dict, path):
    est_dict_str_keys = {}
    for deprel in est_dict:
        est_dict_str_keys[deprel] = {
            f'{r1}->{r2}': count for (r1, r2), count in est_dict[deprel].items()
        }
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(est_dict_str_keys, out, indent=2)


def main(input_file: str, output_dir: str, sample_size: Optional[int] = None):
    trees = conllu2trees(input_file)
    if sample_size:
        trees = random.sample(trees, sample_size)
    estimates = E.get_ml_directionality_estimates(trees)
    file_name = input_file.replace("/", "\\").split("\\")[-1]
    dump_estimates(estimates, f'{output_dir}/{file_name}'
                              f'{".sample_size_" + str(sample_size) if sample_size is not None else ""}'
                              f'.estimates.json')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-i", "--input-file", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)
    argparser.add_argument("-s", "--sample-size", required=False, type=int)

    args = argparser.parse_args()

    main(args.input_file, args.output_dir, args.sample_size)
