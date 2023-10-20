import argparse
import json
import os.path

import conllu
from typing import Dict, List

relevant_deprel = ["nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated",
                   "advcl", "advmod", "aux", "cop", "nmod", "appos", "nummod", "acl", "amod"]


def _get_direction_dict(input_path: str):
    right_direction_dict = {}
    total_dict = {}
    with open(input_path, "r", encoding="utf-8") as input_file:
        for sent in conllu.parse_incr(input_file):
            for node in sent:
                if not isinstance(node['id'], int):
                    continue
                if node['deprel'] not in right_direction_dict:
                    right_direction_dict[node['deprel']] = 0
                if node['deprel'] not in total_dict:
                    total_dict[node['deprel']] = 0
                total_dict[node['deprel']] = total_dict[node['deprel']] + 1
                if node['id'] > node['head']:
                    right_direction_dict[node['deprel']] = right_direction_dict[node['deprel']] + 1

    right_direction_proportion_dict = {}
    for l in total_dict.keys():
        if l in right_direction_dict:
            right_direction_proportion_dict[l] = right_direction_dict[l]/float(total_dict[l])
        else:
            right_direction_proportion_dict[l] = 0

    return right_direction_proportion_dict


def _get_dominant_direction(deprel: str, right_direction_proportion_dict: Dict):
    deprel = deprel.split(":")[0]
    assert ":" not in deprel

    if deprel in right_direction_proportion_dict and deprel in relevant_deprel:
        if right_direction_proportion_dict[deprel] > 0.75:
            return 1
        elif right_direction_proportion_dict[deprel] < 0.25:
            return -1

    return 0


def _get_reordered_mapping_rec(tree: conllu.TokenTree, direction_dict: Dict):
    root_token = {
        "original_position":  int(tree.token['id']),
        "new_position": int(tree.token['id']),
        "deprel": tree.token['deprel']
    }

    if len(tree.children) == 0:
        return [root_token]

    subtrees_nodes = [_get_reordered_mapping_rec(c, direction_dict) for c in tree.children]

    positions = [n['new_position'] for l in subtrees_nodes for n in l] + [tree.token['id']]
    sorted_positions = sorted(positions)
    dominant_directions = [_get_dominant_direction(c.token["deprel"], direction_dict) for c in tree.children]
    directions = [d if d != 0 else 1 if tree.token['id'] < n.token['id'] else -1 for d, n
                  in zip(dominant_directions, tree.children)]
    right_children = [n for i, d in enumerate(directions) for n in subtrees_nodes[i] if d == 1]
    left_children = [n for i, d in enumerate(directions) for n in subtrees_nodes[i] if d == -1]

    right_children = list(sorted(right_children, key=lambda x: x['new_position']))
    left_children = list(sorted(left_children, key=lambda x: x['new_position']))

    nodes = left_children + [root_token] + right_children
    for i, n in enumerate(nodes):
        n['new_position'] = sorted_positions[i]


    return list(sorted(nodes, key=lambda x: x['new_position']))


def reorder_tree(sent: conllu.TokenList, direction_dict: Dict):
    tree = sent.to_tree()
    n = _get_reordered_mapping_rec(tree, direction_dict)

    n_sorted = list(sorted(n, key=lambda x: x['new_position']))

    text = [t['form'] for t in sent if isinstance(t['id'], int)]
    reordered_text = " ".join([text[i['original_position'] - 1] for i in n_sorted if i!=0])

    idx_mapping = {i['original_position']: i['new_position'] for i in n_sorted}
    idx_mapping[0] = 0
    reordered_sent = sent.copy()
    reordered_sent.metadata['text'] = reordered_text
    for n in reordered_sent:
        n['id'] = idx_mapping[n['id']]
        n['head'] = idx_mapping[n['head']]

    reordered_sent.sort(key=lambda x: x["id"])

    return reordered_sent, idx_mapping

def reroder_script():
    argparser = argparse.ArgumentParser(description="Reordering algorithm based on Rasooli and Collins, 2019")
    argparser.add_argument("-i", "--input-path", required=True, help='Path of conllu file to reroder')
    argparser.add_argument("-r", "--reorder-by", required=True, help='Path of a conllu file to reorder stats from')
    argparser.add_argument("-o", "--output-dir", required=True)

    args = argparser.parse_args()
    filename = args.input_path.split("/")[-1]
    estimates_name = args.reorder_by.split("/")[-1]
    output_path = f'{args.output_dir}/{filename}' \
                  f'.reordered_by.{estimates_name}' \
                  f'.conllu'

    direction_dict = _get_direction_dict(args.reorder_by)
    reordered_trees: List[conllu.TokenList] = []

    total = 0
    error = 0
    with open(args.input_path, "r", encoding="utf-8") as input_file:
        for sent in conllu.parse_incr(input_file):
            total = total + 1
            # try:
            n = reorder_tree(sent, direction_dict)
            reordered_trees.append(n[0])
            # except Exception:
            #     error = error + 1


    print(f'{error}/{total}')

    with open(output_path, 'w', encoding='utf-8') as out:
        print('\n\n'.join(t.serialize() for t in reordered_trees), file=out)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-i", "--input-file", required=True)
    argparser.add_argument("-o", "--output-dir", required=True)

    args = argparser.parse_args()

    direction_dict = _get_direction_dict(args.input_file)
    file_name = os.path.basename(args.input_file)
    with open(f'{args.output_dir}/{file_name}.right_direction_prop.json', 'w', encoding='utf-8')as f:
        json.dump(direction_dict, f)





