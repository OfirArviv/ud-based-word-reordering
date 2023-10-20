from functools import cmp_to_key
from typing import List
from copy import deepcopy
from itertools import combinations

import UDLib as U
import z3


def print_tree_w_ordering(UD_tree: U.UDTree, ordering: List[str]):
    for idx in ordering:
        print(UD_tree.nodes[idx].FORM.lower(), end = ' ')
    print()


# def cmp_rel(rel1, rel2, estimate_dict, deprel):
#     if (rel1, rel2) in estimate_dict[deprel]:
#         if estimate_dict[deprel][(rel1, rel2)] > 0.5:
#             return 1
#         else:
#             return -1
#     elif (rel2, rel1) in estimate_dict[deprel]:
#         if estimate_dict[deprel][(rel1, rel2)] > 0.5:
#             return -1
#         else:
#             return 1
#     else:
#         return 0


# def make_cmp_func(UD_tree: U.UDTree, estimate_dict, deprel):
#     def dict_based_comparator(x, y):
#         rel1 = U.get_deprel(x, UD_tree)
#         rel2 = U.get_deprel(y, UD_tree)
#         if rel1 == rel2:
#             return 0
#         result = cmp_rel(rel1, rel2, estimate_dict, deprel)
#         if result != 0:
#             return result
#         elif deprel != 'root':
#             # Fall back on the root dict
#             return cmp_rel(rel1, rel2, estimate_dict, 'root')
#         else:
#             return result
#     return dict_based_comparator


# def reorder_rec(input_tree: U.UDTree, comparator_dict, root_idx, output_arr):
#     root_deprel = U.get_deprel(root_idx, input_tree)
#     print(f"Reordering the expansion of {root_deprel}")

#     nodes = input_tree.get_node_children(root_idx)
#     if not nodes:
#         # Emit the leave
#         output_arr.append(root_idx)
#         return
#     if root_deprel == 'punct':
#         raise ValueError(f"The punctuation node {root_idx} has children!")
#     nodes.append(root_idx)

#     # Leaving this nice hack here for possible future use.
#     # good, bad = [], []
#     # for x in mylist:
#     #     (bad, good)[x in goodvals].append(x)
#     # https://stackoverflow.com/a/12135169/3665134
#     punct_idx = [el for el in nodes if gU.get_deprelet_deprel(el, input_tree) == 'punct']
#     nodes     = [el for el in nodes if gU.get_deprelet_deprel(el, input_tree) != 'punct']

#     # Check if we have learned indices for all relations in
#     # the expansion dict of the root dict, then sort.
#     for node in nodes:
#         if gU.get_deprelet_deprel(node, input_tree) not in comparator_dict[root_deprel]:
#             # Recovery: keep the word in place:
#             # put it after whatever went before it.
#             print(f"Adding index for {U.get_deprel(node, input_tree)}")
#             if node == "1":
#                 min_index = min(comparator_dict[root_deprel].values())
#                 comparator_dict[gU.get_deprelet_deprel(node, input_tree)] = min_index-1
#             else:
#                 prev_node = str(int(node)-1)
#                 prev_idx = comparator_dict[root_deprel][gU.get_deprelet_deprel(prev_node, input_tree)]
#                 comparator_dict[root_deprel][gU.get_deprelet_deprel(node, input_tree)] = prev_idx + 0.1
#     nodes.sort(
#         key=lambda x: comparator_dict[root_deprel][gU.get_deprelet_deprel(x, input_tree)]
#     )

#     # Place punctuation to where it was relative to the root node
#     # before sorting.
#     for p_i in punct_idx:
#         root_pos = nodes.index(root_idx)
#         if int(p_i) < int(root_idx):
#             nodes.insert(root_pos, p_i)
#         else:
#             nodes.insert(root_pos+1, p_i)
#     # Process nodes in their linear order
#     for n in nodes:
#         if n == root_idx:
#             # Emit the root node
#             output_arr.append(root_idx)
#         else:
#             # Process subtrees
#             reorder_rec(input_tree, comparator_dict, n, output_arr)


def solve_expansion(estimates, input_tree, root_deprel, nodes, debug_mode=False):
    # Initialise a solver. Add constraints from the learned dict.
    s = z3.Solver()
    if root_deprel in estimates:
        constraints_dict = estimates[root_deprel]
    else:
        # Fall back on the root dict
        constraints_dict = estimates['root']
        if debug_mode:
            print(f"Falling back on the root dict for {root_deprel}")
        root_deprel = 'root'

    # Add variables for node deprels and corresponding constraints
    # from the estimate dict.
    vars_dict = {}
    all_deprels = set()
    for n in nodes:
        rel = U.get_deprel(n, input_tree)
        all_deprels.add(rel)
        vars_dict[rel] = z3.Ints(n)[0]
        s.add(vars_dict[rel] >= 0)
    for rel1, rel2 in combinations(all_deprels, 2):
        if (rel1, rel2) in constraints_dict:
            if constraints_dict[(rel1, rel2)] > 0:
                s.add(vars_dict[rel1] > vars_dict[rel2])
            else:
                s.add(vars_dict[rel1] < vars_dict[rel2])
        elif (rel2, rel1) in constraints_dict:
            if constraints_dict[(rel2, rel1)] > 0:
                s.add(vars_dict[rel1] < vars_dict[rel2])
            else:
                s.add(vars_dict[rel1] > vars_dict[rel2])
        elif debug_mode:
            print(f"No constraint was learned for {rel1} and {rel2} in the expansion of {root_deprel}")

    # The caller will check if a solution exists.
    return s, vars_dict


def reorder_rec_local(input_tree,
                      estimates,
                      root_idx,
                      new_order,
                      memo,
                      debug_mode=False,
                      error_log=None):
    '''Tries to satisfy the constraints on the relative ordering
    of relations found in the expansion (falls back on root constraints
    when the expansion was not found in the training corpus).'''


    nodes = input_tree.get_node_children(root_idx)
    if not nodes:
        # Emit the leave
        new_order.append(root_idx)
        return

    root_deprel = U.get_deprel(root_idx, input_tree)
    if root_deprel == 'punct':
        raise ValueError(f"The punctuation node {root_idx} has children!")

    nodes.append(root_idx)

    # Remove punctuation
    punct_idx = [el for el in nodes if U.get_deprel(el, input_tree) == 'punct']
    nodes     = [el for el in nodes if U.get_deprel(el, input_tree) != 'punct']

    # Check if the solution was memoized.
    all_deprels = set(
        [U.get_deprel(node, input_tree) for node in nodes]
    )
    deprels_key = tuple(sorted(all_deprels))
    ordering_dict = {}  # For sorting the nodes based on their deprels.
    if (root_deprel, deprels_key) in memo:
        if memo[(root_deprel, deprels_key)] != z3.unsat:
            solution_available = True
            ordering_dict = memo[(root_deprel, deprels_key)]
        else:
            solution_available = False
    else:
        # Try solving if not memoized.
        solver, vars_dict = solve_expansion(estimates, input_tree, root_deprel, nodes)
        # Construct a deprel-ordering dict based on the solution, if it exists.
        if solver.check() != z3.unsat:
            solution_available = True
            model = solver.model()
            for deprel in all_deprels:
                ordering_dict[deprel] = model[vars_dict[deprel]].as_long()
            # Memoize the solution.
            memo[(root_deprel, deprels_key)] = ordering_dict
        else:
            solution_available = False
            memo[(root_deprel, deprels_key)] = z3.unsat

    if solution_available:
        # Reorder the nodes based on the their deprels.
        nodes.sort(key=lambda node: ordering_dict[U.get_deprel(node, input_tree)])
        # Place punctuation to where it was relative to the root node before sorting.
        for p_i in punct_idx:
            root_pos = nodes.index(root_idx)
            if int(p_i) < int(root_idx):
                nodes.insert(root_pos, p_i)
            else:
                nodes.insert(root_pos+1, p_i)
    else:
        if debug_mode:
            if error_log is None:
                raise ValueError("When in debug mode, must pass an error_log counter!")
            error_log[root_deprel] += 1
        # Restore the original order.
        nodes.extend(punct_idx)
        nodes.sort(key=int)

    # Process nodes recursively in the linear order
    for n in nodes:
        if n == root_idx:
            # Emit the root node
            new_order.append(root_idx)
        else:
            # Process subtrees
            reorder_rec_local(input_tree, estimates, n, new_order, memo, debug_mode, error_log)


def switch_keys(tree, key_mapping):
    new_nodes = {
        key_mapping[k]: deepcopy(v) for k, v in tree.nodes.items()
    }
    # Update node IDs
    for k, v in new_nodes.items():
        v.ID = key_mapping[v.ID]
        v.HEAD = key_mapping[v.HEAD]
        new_nodes[k] = v
    new_graph = {
        key_mapping[k]: deepcopy(v) for k, v in tree.graph.items()
    }
    # Update graph edges
    for k, edge_list in new_graph.items():
        for i, new_edge in enumerate(edge_list):
            edge_list[i].head = key_mapping[new_edge.head]
        new_graph[k] = edge_list
    return U.UDTree(tree.id_lines, tree.keys, new_nodes, new_graph)


def reorder(input_tree: U.UDTree, estimates, debug_mode=False, error_log=None):
    '''Reorders the constituents in the input tree in a top-bottom
    fashion based on the relative-ordering estimates. Punctuation
    is placed to the left or to the right from the host depending on where
    it used to be positioned originally.'''

    # Do DFS on the tree, ordering the nodes at each level, traversing
    # the children from left to right, and emitting at the lowest level.
    new_order = []
    memo = {}  # memoize the solutions for particular combinations
    root_idx = input_tree.get_real_root()
    reorder_rec_local(input_tree,
                      estimates,
                      root_idx,
                      new_order,
                      memo,
                      debug_mode,
                      error_log)

    # Transform the tree based on the new ordering.
    # We simply map the old keys to their new positions
    # and then switch the keys.
    new_tree = U.purge_dotted_nodes(input_tree)
    assert len(new_order) == len(new_tree.keys)
    idx_map = {
        new_idx: old_idx for old_idx, new_idx in zip(new_tree.keys, new_order)
    }
    # Map the root to itself
    idx_map['0'] = '0'
    new_tree = switch_keys(new_tree, idx_map)

    return new_tree