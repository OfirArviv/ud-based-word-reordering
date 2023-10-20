from copy import deepcopy
from typing import Dict, Tuple, List

from itertools import combinations, permutations

import gurobipy as gb
from gurobipy import GRB

import z3

import reordering_package.huji_ud_reordering.UDLib as U


def reorder_tree(
    tree: U.UDTree,
    estimates: Dict[str,Dict[Tuple[str,str],int]],
    separate_neg=True,
    with_gurobi=True,
    preference_threshold=0.8
):
    new_order = []
    root_idx = tree.get_real_root()
    if with_gurobi:
        reorder_tree_rec(tree,
                         estimates,
                         root_idx,
                         new_order,
                         separate_neg,
                         preference_threshold)
    else:
        reorder_tree_rec_no_gurobi(tree,
                                   estimates,
                                   root_idx,
                                   new_order,
                                   separate_neg)

    # Transform the tree based on the new ordering.
    # We map the old keys to their new positions
    # and then switch the keys.
    new_tree = U.purge_dotted_nodes(tree)
    assert len(new_order) == len(new_tree.keys)
    idx_map = {
        new_idx: old_idx for old_idx, new_idx in zip(new_tree.keys, new_order)
    }
    # Map the root to itself
    idx_map['0'] = '0'
    new_tree = switch_keys(new_tree, idx_map)

    return new_tree, idx_map


def reorder_tree_rec_no_gurobi(tree: U.UDTree,
                               estimates: Dict[str,Dict[Tuple[str,str],int]],
                               root_idx: str,
                               new_order: List[str],
                               separate_neg=True):
    nodes = tree.get_node_children(root_idx)
    if not nodes:
        # Emit the leave
        new_order.append(root_idx)
        return

    root_deprel = U.get_deprel(root_idx, tree, separate_neg)
    if root_deprel == 'punct':
        raise ValueError(f"The punctuation node {root_idx} has children!")

    nodes.append(root_idx)
    word_idx, punct_idx = remove_punctuation(nodes, tree)    

    if root_deprel in estimates and len(word_idx) > 1 and \
        len(set([U.get_deprel(el, tree, separate_neg) for el in word_idx])) > 1:
        pairwise_orderings = get_empirical_pairwise_ordering(
            word_idx, tree, estimates[root_deprel], separate_neg
        )

        # Try to apply SMT.
        indices_dict = ordering2indices(pairwise_orderings)
        if indices_dict is not None:
            word_idx.sort(key=lambda word: indices_dict[U.get_deprel(word, tree, separate_neg)])

            # Place the punctuation back to where it was relative to the root node.
            nodes = [el for el in word_idx]
            for p_i in punct_idx:
                root_pos = nodes.index(root_idx)
                if int(p_i) < int(root_idx):
                    nodes.insert(root_pos, p_i)
                else:
                    nodes.insert(root_pos+1, p_i)

    # Process the nodes recursively in the linear order.
    for n in nodes:
        if n == root_idx:
            # Emit the root node.
            new_order.append(root_idx)
        else:
            # Process subtrees.
            reorder_tree_rec_no_gurobi(tree, estimates, n, new_order, separate_neg)


def get_empirical_pairwise_ordering(
    nodes: List[str],
    tree: U.UDTree,
    estimates: Dict[str,Dict[Tuple[str,str],int]],
    separate_neg=True
):
    '''
    Converts estimates of relative frequency of different
    pairwise orderings of deprels into a bunch of binary values.
    A pair of values for root->aux/aux->root, etc., cannot be straightforwardly
    contradictory (both values cannot be set to 1), but it cannot be
    uninformative (both values set to 0) when not enough dataset were gathered.
    Moreover, there is no guarantee that there are no loops/transitivity
    conflicts between variables with different deprels. Use the "optimal"
    function to ensure that.
    '''
    all_deprels = set([U.get_deprel(el, tree, separate_neg) for el in nodes])

    # Construct the variables and set their values at once
    # based on relative frequencies.
    var_dict = {}
    for rel1, rel2 in combinations(all_deprels, 2):
        key1 = f'{rel1}->{rel2}'
        key2 = f'{rel2}->{rel1}'
        freq1 = estimates.get(key1, 0);
        freq2 = estimates.get(key2, 0);
        # Both variables cannot be set to 1, but both
        # can be set to 0, thus leading to an unconstrained ordering.
        var_dict[key1] = 1 if freq1 > freq2 else 0
        var_dict[key2] = 1 if freq2 > freq1 else 0

    # Pass the constructed variables with their values to the SMT solver.
    return [(key, val) for key, val in var_dict.items()]


def reorder_tree_rec(tree: U.UDTree,
                     estimates: Dict[str,Dict[Tuple[str,str],int]],
                     root_idx: str,
                     new_order: List[str],
                     separate_neg=True,
                     preference_threshold=0.8):
    nodes = tree.get_node_children(root_idx)
    if not nodes:
        # Emit the leave
        new_order.append(root_idx)
        return

    root_deprel = U.get_deprel(root_idx, tree, separate_neg)
    if root_deprel == 'punct':
        raise ValueError(f"The punctuation node {root_idx} has children!")

    nodes.append(root_idx)
    word_idx, punct_idx = remove_punctuation(nodes, tree)

    if root_deprel in estimates and \
        len(word_idx) > 1 and \
        len(set([U.get_deprel(el, tree, separate_neg) for el in word_idx])) > 1:
        # The wanky stuff.
        # MIP program.
        pairwise_orderings = get_optimal_pairwise_ordering(
            word_idx, tree, estimates[root_deprel], preference_threshold, separate_neg
        )
        # SMT solver.
        indices_dict = ordering2indices(pairwise_orderings)
        if indices_dict is None:
            raise ValueError(f'z3 failed to solve the expansion of {root_idx}.')

        # The actual reordering.
        word_idx.sort(key=lambda word: indices_dict[U.get_deprel(word, tree, separate_neg)])

    # Place the punctuation back to where it was relative to the root node.
    nodes = [el for el in word_idx]
    for p_i in punct_idx:
        root_pos = nodes.index(root_idx)
        if int(p_i) < int(root_idx):
            nodes.insert(root_pos, p_i)
        else:
            nodes.insert(root_pos+1, p_i)

    # Process the nodes recursively in the linear order.
    for n in nodes:
        if n == root_idx:
            # Emit the root node.
            new_order.append(root_idx)
        else:
            # Process subtrees.
            reorder_tree_rec(tree, estimates, n, new_order, separate_neg, preference_threshold)


def get_optimal_pairwise_ordering(
    nodes: List[str],
    tree: U.UDTree,
    estimates: Dict[str,Dict[Tuple[str,str],int]],
    preference_threshold=0.8,
    separate_neg=True
):
    '''
    Computes an optimal conflict-free pairwise ordering of deprels in
    the input sentence based on the learned preferences. If there are
    gaps in the learned estimates or those are weak, input-sentence
    pairwise order is used a source of constraints.
    '''
    try:
        m = gb.Model('ordering_ilp')
        m.setParam('OutputFlag', 0)  # Kill the output.

        # Create variables for all possible pairwise orderings
        # of deprels in the input expansion.
        all_deprels = set([U.get_deprel(el, tree, separate_neg) for el in nodes])
        var_dict = {}
        for rel1, rel2 in combinations(all_deprels, 2):
            key1 = f'{rel1}->{rel2}'  # i,j
            key2 = f'{rel2}->{rel1}'  # j,i
            var_dict[key1] = m.addVar(vtype=GRB.BINARY, name=key1)
            var_dict[key2] = m.addVar(vtype=GRB.BINARY, name=key2)

            # Limit the edges to one direction
            constr_name = f'{key1} + {key2} == 1'
            m.addConstr(var_dict[key1] + var_dict[key2] == 1, constr_name)

        # Prohibit 3-cycles in the complete graph with directed edges.
        for rel_triple in combinations(all_deprels, 3):
            for rel1, rel2, rel3 in permutations(rel_triple):
                key1 = f'{rel1}->{rel2}'  # i,j
                key2 = f'{rel2}->{rel3}'  # j,k
                key3 = f'{rel3}->{rel1}'  # k,i
                constr_name = f'{key1} + {key2} + {key3} <= 2'
                m.addConstr(
                    var_dict[key1] + var_dict[key2] + var_dict[key3] <= 2,
                    constr_name
                )

        # Construct the objective.
        # Omri:
        # My original thinking was that we give a high weight to clear
        # preferences (where more than 80% are towards one direction), say that
        # selecting such an edge will get a score of 10^5. Then for pairs where
        # there is no edge in one direction or another because 0.2 < p < 0.8,
        # then use the original order with weight 1.
        linear_expression = 0
        for rel1, rel2 in combinations(all_deprels, 2):
            c_ij = c_ji = 0
            direct = estimates.get(f'{rel1}->{rel2}', 0)
            reverse = estimates.get(f'{rel2}->{rel1}', 0)
            if direct > 0 and reverse > 0:
                # We have estimates in both directions, need to check
                # if there is a clear preference.
                if direct / (direct+reverse) >= preference_threshold:
                    c_ij = 10**5
                elif reverse / (direct+reverse) >= preference_threshold:
                    c_ji = 10**5
                else:
                    # Use the input-expansion order.
                    # Look for all combinations of nodes until
                    # the right one is found.
                    for n1, n2 in combinations(nodes, 2):
                        r1 = U.get_deprel(n1, tree, separate_neg)
                        r2 = U.get_deprel(n2, tree, separate_neg)
                        if r1 == rel1 and r2 == rel2:
                            c_ij = 1
                            break
                        elif r1 == rel2 and r2 == rel1:
                            c_ji = 1
                            break
            elif direct > 0:
                c_ij = 10**5
            elif reverse > 0:
                c_ji = 10**5
            else:
                # No estimates were learned. Use the input expansion.
                for n1, n2 in combinations(nodes, 2):
                    r1 = U.get_deprel(n1, tree, separate_neg)
                    r2 = U.get_deprel(n2, tree, separate_neg)
                    if r1 == rel1 and r2 == rel2:
                        c_ij = 1
                        break
                    elif r1 == rel2 and r2 == rel1:
                        c_ji = 1
                        break
            if c_ij > 0:
                linear_expression += var_dict[f'{rel1}->{rel2}'] * c_ij
            if c_ji > 0:
                linear_expression += var_dict[f'{rel2}->{rel1}'] * c_ji

        m.setObjective(linear_expression, GRB.MAXIMIZE)
        m.optimize()

        result = [(var.varName, var.x) for var in m.getVars()]

        return result

    except gb.GurobiError as e:
        print(f'Error code {e.errno}: {e}.')

    except AttributeError as e:
        print(f'Encountered an attribute error: {e}.')


def ordering2indices(ordering_vars):
    s = z3.Solver()

    # Extract SMT variables and constraints from the MIP solver output.
    var_dict = {}
    all_deprels = set()
    for var in ordering_vars:
        varName, x = var
        rel1, rel2 = varName.split('->')
        all_deprels.add(rel1)
        all_deprels.add(rel2)
        if rel1 not in var_dict:
            var_dict[rel1] = z3.Ints(rel1)[0]
            s.add(var_dict[rel1] >= 0)
        if rel2 not in var_dict:
            var_dict[rel2] = z3.Ints(rel2)[0]
            s.add(var_dict[rel2] >= 0)
        # We only take less-than constraints.
        # Greater-then constraints are their duals.
        if x < 0.5:
            s.add(var_dict[rel1] > var_dict[rel2])

    # Should be solvable!
    if s.check() == z3.unsat:
        return None
    else:
        model = s.model()
        return {
            deprel: model[var_dict[deprel]].as_long() for deprel in all_deprels
        }


def remove_punctuation(nodes, tree):
    punct_idx = [el for el in nodes if U.get_deprel(el, tree) == 'punct']
    word_idx  = [el for el in nodes if U.get_deprel(el, tree) != 'punct']
    return word_idx, punct_idx


def switch_keys(tree: U.UDTree, key_mapping: Dict[str,str]):
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
