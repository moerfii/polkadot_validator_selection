import json

import numpy as np
import cvxpy as cp
import pandas as pd
import scipy as sp

def solve_validator_selection(dataframe):
    """
    This function is supposed to solve the total_bond distribution amongst the validators elected.
    It aims to maximise the minimally staked validator.
    It ensures that the total_bond does not exceed the sum of the stakes attributed to the individual validators.
    :param dataframe:
    :return:
    """
    validator_names = dataframe["validator"].unique()
    nominator_names = dataframe["nominator"].unique()

    # Create dictionary to map nominator and validator names to their respective row and column indices
    nominator_indices = {name: i for i, name in enumerate(nominator_names)}
    validator_indices = {name: i for i, name in enumerate(validator_names)}

    # Create matrix of voter preferences
    voter_preferences = np.zeros((len(nominator_names), len(validator_names)))
    for i, row in dataframe.iterrows():
        voter_preferences[nominator_indices[row["nominator"]], validator_indices[row["validator"]]] = 1

    # calculate proportional bond by grouping the total_bond by nominator and getting the mean
    proportional_bonds = dataframe.groupby("nominator")["total_bond"].mean()

    # set the 1's in the matrix to the corresponding values in the vector
    matrix = voter_preferences * proportional_bonds.values[:, None]
    non_zero_mask = (matrix != 0)

    """    max_abs = np.max(np.abs(matrix))

    # scale the matrix by dividing each element by the maximum absolute value
    matrix = matrix / max_abs
    """
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)



    # create a mask for non-zero elements in the matrix


    # create the variables to be optimised, it should be in the shape of the matrix
    x = cp.Variable(matrix.shape)

    # create the constraints
    # the values cannot be negative
    # the sum of each row must remain the same as defined in the matrix
    constraints = [
        x >= 0,
        cp.sum(x, axis=1) == matrix.sum(axis=1),
        x[~non_zero_mask] == 0]

    # create the objective function
    # the objective function is to maximise the minimum sum of the columns
    objective = cp.Maximize(cp.min(cp.sum(x, axis=0)))

    # create the problem
    problem = cp.Problem(objective, constraints)

    # solve the problem
    problem.solve(verbose=True, solver=cp.SCS, max_iters=10000)

    # print the results
    print("The optimal value is", problem.value)
    print("A solution x is")
    print(x.value)
    dataframe = x.value
    pd.DataFrame(x.value).to_csv("../data/model_2/591_max_min_stake.csv")
    print("A dual solution corresponding to the inequality constraints is")
    print(constraints[0].dual_value)









if __name__ == "__main__":

    dataframe = pd.read_csv("../data_collection/data/model_2/df_bond_distribution_testing_0.csv")
    dataframe = dataframe[dataframe["era"] == 591]
    """
    # example dataframe
    dataframe = pd.DataFrame(   {"nominator": ["A", "A", "B", "B", "B", "B", "C", "A", "C"],
                                    "validator": ["X", "Y", "Z", "X", "Y", "X", "X", "Y", "Z"],
                                    "total_bond": [10, 20, 30, 70, 0, 20, 30, 0, 30]})
    """
    solve_validator_selection(dataframe)