import json

import numpy as np
import cvxpy as cp
import pandas as pd
import scipy as sp
from sklearn import preprocessing


def solve_validator_selection(snapshot, winners, era):
    print(f"Starting era {era}")
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

    nominator_mapping = {}
    nominator_index = 0
    validator_mapping = {}
    validator_index = 0
    # Create matrix of voter preferences
    voter_preferences = np.zeros((len(nominator_names), len(validator_names)))
    for row in voters:
        length_active_validators = sum([1 if validator in validator_names else 0 for validator in row[2]])
        if length_active_validators == 0:
            continue
        nominator_mapping[nominator_index] = row[0]
        nominator_index += 1
        proportional_bond = row[1] / length_active_validators
        for validator in row[2]:
            try:
                voter_preferences[
                    nominator_indices[row[0]], validator_indices[validator]
                ] = proportional_bond
                if validator not in validator_mapping:
                    validator_mapping[validator] = validator_index
                    validator_index += 1
            except KeyError:
                print(f"validator: {validator} is not in available targets")
                pass

    voter_preferences = voter_preferences[
        ~np.all(voter_preferences == 0, axis=1)
    ]

    rev_validator_mapping = {v: k for k, v in validator_mapping.items()}
    # change voter preference into dataframe and rename column and rows based on mappings
    voter_preferences = pd.DataFrame(voter_preferences)
    voter_preferences = voter_preferences.rename(columns=rev_validator_mapping)
    voter_preferences = voter_preferences.rename(index=nominator_mapping)
    # save voter preferences to json
    voter_preferences.to_csv(f"../data/model_2/{era}_voter_preferences.csv")

    """  #
    save mappings to json
    with open(f"../data/model_2/{era}_nominator_mapping.json", "w") as jsonfile:
        json.dump(nominator_mapping, jsonfile)
    rev_validator_mapping = {v: k for k, v in validator_mapping.items()}
    with open(f"../data/model_2/{era}_validator_mapping.json", "w") as jsonfile:
        json.dump(rev_validator_mapping, jsonfile)
        
    """
    """    # drop rows with all zeros
    voter_preferences = voter_preferences[
        ~np.all(voter_preferences == 0, axis=1)
    ]

    # previous distribution
    prior_to_optimisation = voter_preferences.sum(axis=0)

    non_zero_mask = voter_preferences != 0

    scaler = 1/voter_preferences.mean()
    voter_preferences = voter_preferences * scaler

     # create the variables to be optimised, it should be in the shape of the matrix
    x = cp.Variable(voter_preferences.shape, nonneg=True, name="x")

    # create the constraints
    # the values cannot be negative
    # the sum of each row must remain the same as defined in the matrix
    # only 297 validators columns can have a non-zero sum
    constraints = [
        cp.sum(x, axis=1) == voter_preferences.sum(axis=1),
        x[~non_zero_mask] == 0
    ]

    # create the objective function
    # the objective function is to maximise the minimum sum of the columns
    objective = cp.Maximize(cp.min(cp.sum(x, axis=0)))

    # create the problem
    problem = cp.Problem(objective, constraints)

    # solve the problem
    problem.solve(solver=cp.SCS, verbose=True, max_iters=5000)


    # print the results
    print("The optimal value is", problem.value)
    dataframe = x.value / scaler
    pd.DataFrame(dataframe).to_csv(f"../data/model_2/{era}_max_min_stake.csv")
    print(f"Finished era {era}")"""


if __name__ == "__main__":
    # 986 inaccurate, reached max iterations
    # 990 inaccurate, reached max iterations
    # 991 inaccurate, reached max iterations
    # 993 inaccurate, reached max iterations
    eras = [985, 986, 987, 988, 989, 990, 991, 992, 993, 994]
    
    for era in eras:
        with open(f"../data/snapshot_data/{era}_snapshot.json", "r") as f:
            snapshot = json.load(f)

        with open(f"../data/calculated_solutions_data/{era}_winners.json", "r") as f:
            winners = json.load(f)

        solve_validator_selection(snapshot, winners, era)

    """
    dataframe = pd.read_csv("../data/model_2/1_max_min_stake.csv", index_col=0)
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

    solve_validator_selection(snapshot, winners, 1)

    """