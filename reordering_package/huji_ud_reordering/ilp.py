from itertools import combinations, permutations
from collections import Counter

import gurobipy as gb
from gurobipy import GRB

import z3


def get_pairwise_ordering(training_set_constraints: Counter):
    '''
    Solves an integer program and returns a non-loopy ordering
    of deprels in an expansion to be converted into indices using
    an SMT solver.
    '''
    try:
        m = gb.Model('ordering_ilp')

        # Create variables for all possible pairwise orderings.
        all_deprels = set()
        for r1, r2 in training_set_constraints:
            all_deprels.add(r1)
            all_deprels.add(r2)
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
        linear_expression = 0
        for rel1, rel2 in combinations(all_deprels, 2):
            key1 = f'{rel1}->{rel2}'  # i,j
            # Add 1 to eliminate 0 weights.
            c_ij = 1.0 + training_set_constraints[key1]
            linear_expression += var_dict[key1] * c_ij
            key2 = f'{rel2}->{rel1}'  # j,i
            c_ji = 1.0 + training_set_constraints[key2]
            linear_expression += var_dict[key2] * c_ji

        m.setObjective(linear_expression, GRB.MAXIMIZE)
        m.optimize()

        return [(var.varName, var.x) for var in m.getVars()]

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
