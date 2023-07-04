import json

import cvxpy as cp
import numpy as np
import pandas as pd


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
        length_active_validators = sum(
            [1 if validator in validator_names else 0 for validator in row[2]]
        )
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
    voter_preferences = voter_preferences[~np.all(voter_preferences == 0, axis=1)]

    # previous distribution
    # prior_to_optimisation = voter_preferences.sum(axis=0)

    non_zero_mask = voter_preferences != 0
    # create the variables to be optimised, it should be in the shape of the matrix
    x = cp.Variable(voter_preferences.shape, nonneg=True)

    # create the constraints
    # the values cannot be negative
    # the sum of each row must remain the same as defined in the matrix
    # only 297 validators columns can have a non-zero sum
    constraints = [
        cp.sum(x, axis=1) == voter_preferences.sum(axis=1),
        x[~non_zero_mask] == 0,
    ]

    # create the objective function
    # the objective function is to maximise the minimum sum of the columns
    objective = cp.Maximize(cp.min(cp.sum(x, axis=0)))

    # create the problem
    problem = cp.Problem(objective, constraints)

    # solve the problem
    problem.solve(verbose=True, solver=cp.SCIP, max_iters=10000)

    # posterior_to_optimisation = (x.value).sum(axis=0)

    # print the results
    print("The optimal value is", problem.value)
    print("A solution x is")
    print(x.value)
    pd.DataFrame(x.value).to_csv("../data/model_2/591_max_min_stake.csv")
    print("A dual solution corresponding to the inequality constraints is")
    print(constraints[0].dual_value)


if __name__ == "__main__":

    with open("../data_collection/data/snapshot_data/994_snapshot.json", "r") as f:
        snapshot = json.load(f)

    with open(
        "../data_collection/data/calculated_solutions_data/994_winners.json", "r"
    ) as f:
        winners = json.load(f)

    """snapshot = {
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
    ]"""

    solve_validator_selection(snapshot, winners)
