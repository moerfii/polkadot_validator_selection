import json
import time

import numpy as np
import cvxpy as cp
import pandas as pd


def proportional_adjust(dataframe):

    difference = (
        dataframe["proportional_bond"].sum()
        - dataframe.groupby("validator")["expected_sum_stake"].first().sum()
    )

    ratio = (
        dataframe.groupby("validator")["expected_sum_stake"].first()
        / dataframe.groupby("validator")["expected_sum_stake"].first().sum()
    )
    dataframe["ratio"] = dataframe["validator"].map(ratio)
    dataframe["expected_sum_stake"] += dataframe["ratio"] * difference
    difference = (
        dataframe["proportional_bond"].sum()
        - dataframe.groupby("validator")["expected_sum_stake"].first().sum()
    )

    if difference != 0:
        print("Difference is not zero")
        print(f"Difference: {difference}")
        dataframe.loc[0, "expected_sum_stake"] += difference

    difference = (
        dataframe["proportional_bond"].sum()
        - dataframe.groupby("validator")["expected_sum_stake"].first().sum()
    )

    return dataframe


def minimise_diagonal_variance(x):
    n = x.shape[1]  # number of columns
    covariances = (
        cp.diag(cp.matmul(cp.transpose(x), x)) - cp.square(cp.sum(x, axis=0)) / n
    )
    return cp.Minimize(cp.sum(covariances))


def minimise_mad(x):
    return cp.Minimize(mad(cp.sum(x, axis=0)))


def minimise_var(x):
    return cp.Minimize(variance(cp.sum(x, axis=0)))


def mad(X):
    median = mean(X)
    return cp.sum(cp.abs(X - median)) / X.size


def mean(x):
    return cp.sum(x) / x.size


def variance(X: cp.Variable, mode="unbiased"):
    if mode == "unbiased":
        scale = X.size - 1
    elif mode == "mle":
        scale = X.size
    else:
        raise ValueError("unknown mode: " + str(mode))
    return cp.sum_squares(X - mean(X)) / scale


def maximise_min(x):
    return cp.Maximize(cp.min(cp.sum(x, axis=0)))


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
        length_active_validators = sum(
            [1 if validator in validator_names else 0 for validator in row[2]]
        )
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
                # print(f"validator: {validator} is not in available targets")
                pass

    """
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
    """

    # drop rows with all zeros
    voter_preferences = voter_preferences[~np.all(voter_preferences == 0, axis=1)]

    # voter_preferences = csr_matrix(voter_preferences)

    # previous distribution
    voter_preferences.sum(axis=0)

    non_zero_mask = voter_preferences != 0

    scaler = 1 / voter_preferences.mean()
    voter_preferences = voter_preferences * scaler

    # create the variables to be optimised, it should be in the shape of the matrix
    x = cp.Variable(voter_preferences.shape, nonneg=True, name="x")

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
    # objective = cp.Maximize(cp.min(cp.sum(x, axis=0)))
    # objective = maximise_min(x)
    # covariance_matrix = np.cov(voter_preferences.T)

    # new objective function is to minimise the variance of the sum of the columns
    objective = minimise_var(x)

    # create the problem
    problem = cp.Problem(objective, constraints)

    # solve the problem # SCS for max min stake, ECOS for min var stake
    problem.solve(
        solver=cp.SCS,
        verbose=True,
        max_iters=5000,
    )

    # print variance of voter preferences
    print(
        f"Variance of voter preferences: {np.var((voter_preferences /scaler).sum(axis=0))}"
    )
    print(f"Variance of optimised solution: {np.var((x.value / scaler).sum(axis=0))}")

    # print the results
    print("The optimal value is", problem.value)
    dataframe = x.value / scaler
    pd.DataFrame(dataframe).to_csv(f"../data/model_2/{era}_min_var_stake.csv")
    print(f"Finished era {era}")


def solve_stake_distribution(dataframe):
    start_time = time.time()

    # expected_sum_stake = dataframe.groupby("validator")['expected_sum_stake'].nth(0)

    matrix_dataframe = dataframe[["nominator", "validator", "proportional_bond"]]
    matrix_dataframe = matrix_dataframe.pivot(
        index="nominator", columns="validator", values="proportional_bond"
    )
    matrix_dataframe = matrix_dataframe.fillna(0)

    non_zero_mask = matrix_dataframe != 0

    scaler = 1 / matrix_dataframe.values.mean()
    matrix_dataframe = matrix_dataframe * scaler
    # scaler_2 = 1 / expected_sum_stake.mean()
    # expected_sum_stake = expected_sum_stake * scaler_2

    x = cp.Variable(matrix_dataframe.shape, nonneg=True, name="x")

    # create the constraints: the values cannot be negative, the sum of each row must remain the same as defined in the matrix, and the columns must sum approximately to the expected sum stake
    constraints = [
        cp.sum(x, axis=1) == matrix_dataframe.sum(axis=1),
        x[~non_zero_mask] == 0,
        # cp.sum(x, axis=0) >= expected_sum_stake
    ]

    # create the objective function
    # the objective function is to maximise the minimum sum of the columns
    objective = cp.Maximize(cp.min(cp.sum(x, axis=0)))

    # create the problem
    problem = cp.Problem(objective, constraints)

    problem.solve(
        solver=cp.SCS,
        verbose=True,
        max_iters=10000,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    dataframe = x.value / scaler
    dataframe = pd.DataFrame(dataframe)
    sums = dataframe.sum(axis=0)
    print(f"era: {era}")
    print(f"min:   {np.min(sums)}")
    print(f"sum:   {np.sum(sums)}")
    print(f"var:   {np.var(sums)}")
    dataframe.to_csv(f"../data/solved_solutions/{era}_solved.csv")
    matrix_dataframe.to_csv(f"../data/solved_solutions/{era}_index.csv")


if __name__ == "__main__":

    for era in range(1000, 1010):

        dataframe = pd.read_csv(
            f"../data/intermediate_results/{era}_model_2_predictions.csv"
        )
        dataframe_distribution = pd.read_csv(
            f"../data/processed_data/model_2_data_Xtest_{era}.csv"
        )

        print(len(dataframe_distribution.groupby("validator")
            .nth(0)))

        top_validators = (
            dataframe_distribution.groupby("validator")
            .nth(0)
            .sort_values(by=["probability_of_selection"], ascending=False)["validator"]
            .head(297)
        )


        top_dataframe = dataframe_distribution[
            dataframe_distribution["validator"].isin(top_validators)
        ]

        top_dataframe.loc[:, "proportional_bond"] = top_dataframe.groupby("nominator")[
            "total_bond"
        ].transform(lambda x: x / x.count())
        """
        top_validators = top_dataframe.groupby("validator").sum().sort_values(by="proportional_bond", ascending=False).reset_index()['validator'].head(297)
    
        topest_dataframe = dataframe_distribution[dataframe_distribution['validator'].isin(top_validators)]
    
    
        topest_dataframe.loc[:, 'proportional_bond'] = topest_dataframe.groupby("nominator")['total_bond'].transform(lambda x: x / x.count())
    
        """

        path = f"../data/calculated_solutions_data/{era}_winners.json"
        with open(path) as f:
            winners = json.load(f)

        winners = [x[0] for x in winners]

        diff = set(winners).difference(set(top_validators))

        sum = top_dataframe.groupby("validator")["proportional_bond"].sum().sum()

        # 155kd7ngDyNnYaEnBBd1wESpUqfN4GmzGL4XgjkCUQpjCrFh
        # 16G8NDzxUeUbGiw2bFX3Wy7JwNEJz9U8B1smCFqqe4GPZbdN

        solve_stake_distribution(top_dataframe)
    # era 950
    [14859076235856999, 5714282646561900371, 80893769691251710879661102727168]
    [14856922854405486, 5714518580240882000, 117768636167906680000000000000000]

    # era 949
    [13947290473163057, 5699917970780130445, 81074286507278729908411005140992]
    [5708617365843472]
    5.708557258079818e18

    # solve_stake_distribution(dataframe_distribution)
    """
    #solve_stake_distribution(adjusted_df)

    # example dataframe for stake distribution
    example_dataframe = [
        ["nominator_1", "validator_1", 40, 220],
        ["nominator_1", "validator_2", 40, 270],
        ["nominator_1", "validator_3", 40, 300],
        ["nominator_2", "validator_1", 90, 220],
        ["nominator_2", "validator_2", 90, 270],
        ["nominator_2", "validator_3", 90, 190],
        ["nominator_3", "validator_1", 200, 220],
        ["nominator_3", "validator_2", 200, 270],

    ]

    example_dataframe = pd.DataFrame(example_dataframe, columns=["nominator", "validator", "proportional_bond", "expected_sum_stake"])
    example_dataframe = proportional_adjust(example_dataframe)

    solve_stake_distribution(example_dataframe)"""

    """
    # max min stake
    # 986 inaccurate, reached max iterations
    # 990 inaccurate, reached max iterations
    # 991 inaccurate, reached max iterations
    # 993 inaccurate, reached max iterations
    # min var stake

    eras = [985, 986, 987, 988, 989, 990, 991, 992, 993, 994]

    for era in eras:
        with open(f"../data/snapshot_data/{era}_snapshot.json", "r") as f:
            snapshot = json.load(f)

        with open(
            f"../data/calculated_solutions_data/{era}_winners.json", "r"
        ) as f:
            winners = json.load(f)

        solve_validator_selection(snapshot, winners, era)"""

    """
    #dataframe = pd.read_csv("../data/model_2/1_max_min_stake.csv", index_col=0)
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
