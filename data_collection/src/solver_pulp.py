import json
import numpy as np
import pandas as pd
from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, value
from sklearn import preprocessing


def solve_validator_selection(snapshot, winners):
    """
    This function is supposed to solve the total_bond distribution amongst the validators elected.
    It aims to maximise the minimally staked validator.
    It ensures that the total_bond does not exceed the sum of the stakes attributed to the individual validators.
    :param snapshot:
    :return:
    """

    voters = snapshot["voters"]
    nominator_names = [voter[0] for voter in voters]
    validator_names = [winner[0] for winner in winners]

    # Create dictionary to map nominator and validator names to their respective row and column indices
    nominator_indices = {name: i for i, name in enumerate(nominator_names)}
    validator_indices = {name: i for i, name in enumerate(validator_names)}

    # Create matrix of voter preferences
    voter_preferences = np.zeros((len(nominator_names), len(validator_names)))
    for row in voters:
        length_active_validators = sum([1 if validator in validator_names else 0 for validator in row[2]])
        if length_active_validators == 0:
            continue
        proportional_bond = row[1] / length_active_validators
        for validator in row[2]:
            try:
                voter_preferences[
                    nominator_indices[row[0]], validator_indices[validator]
                ] = proportional_bond
            except KeyError:
                pass

    # drop rows with all zeros
    voter_preferences = voter_preferences[
        ~np.all(voter_preferences == 0, axis=1)
    ]

    # previous distribution
    prior_to_optimisation = voter_preferences.sum(axis=0)

    zero_mask = voter_preferences == 0
    # create the variables to be optimised, it should be in the shape of the matrix
    x = LpVariable.dicts("x", ((i, j) for i in range(len(nominator_names)) for j in range(len(validator_names))),
                         lowBound=0, cat="Integer")

    # create the problem
    prob = LpProblem("Validator_Selection", LpMaximize)

    # add the objective function, which is to maximise the minimum sum of the individual columns
    # create the objective function
    # the objective function is to maximise the minimum sum of the columns
    min_col_sums = [lpSum([x[i, j] for i in range(voter_preferences.shape[0])]) for j in range(voter_preferences.shape[1])]
    prob += lpSum(min_col_sums)
    print(prob)



    # the sum of each row in voter_preferences must remain
    for i in range(len(nominator_names)):
        prob += lpSum([x[(i, j)] for j in range(len(validator_names))]) == voter_preferences[i, :].sum()


    # add constraint that the values defined in the zero_mask must remain zero
    for i in range(len(nominator_names)):
        for j in range(len(validator_names)):
            if zero_mask[i, j]:
                prob += x[(i, j)] == 0

    # add constraint that the non-zero values must be bigger or equal to zero
    for i in range(len(nominator_names)):
        for j in range(len(validator_names)):
            if not zero_mask[i, j]:
                prob += x[(i, j)] >= 0

    # solve the problem
    prob.solve()


    posterior_to_optimisation = np.array(
        [x[(i, j)].varValue for i in range(len(nominator_names)) for j in range(len(validator_names))]).reshape(
        (len(nominator_names), len(validator_names)))

    # print the results
    print("Status:", LpProblem.status[prob.status])
    print("Objective:", value(prob.objective))
    print("Optimal Solution:")



if __name__ == "__main__":

    with open("../data/snapshot_data/994_snapshot.json", "r") as f:
        snapshot = json.load(f)

    with open("../data/calculated_solutions_data/994_winners.json", "r") as f:
        winners = json.load(f)

    snapshot = {
        "voters": [
            ["voter1", 100, ["candidate1", "candidate2", "candidate3", "candidate4"]],
            ["voter2", 50, ["candidate2", "candidate3"]],
            ["voter3", 200, ["candidate4", "candidate5"]],
            ["voter4", 150, ["candidate1", "candidate5"]],
            ["voter5", 50, ["candidate1", "candidate2"]],
        ]
    }

    winners = [
        ["candidate1", 0.5],
        ["candidate2", 0.3],
        ["candidate3", 0.2],
        ["candidate4", 0.1],
        ["candidate5", 0.1],
    ]

    solve_validator_selection(snapshot, winners)