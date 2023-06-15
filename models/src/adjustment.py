import time

import numpy as np
import multiprocessing
import cvxpy as cp
import pandas as pd


"""
ONLY WORKS WITH COMPLETE SINGULAR ERAS
get test_dataframe with predictions made by model
go through nominators, add up total stakes and adjust to 100%
implement various adjustment strategies
"""


class AdjustmentTool:
    def __init__(self, dataframe=None):
        self.adjusted_dataframe = None
        self.dataframe = dataframe

    def apply_even_split_strategy(self, nominator, dataframe):
        nominator_df = dataframe.loc[
            dataframe["nominator"] == nominator
        ].reset_index(drop=True)
        # if the nominator_df consists of only one row, we simply set the prediction equal to the total bond
        if len(nominator_df["prediction"]) == 1:
            nominator_df.loc[0, "prediction"] = nominator_df.loc[
                0, "total_bond"
            ]
            return nominator_df

        # count how many 0 predictions there are
        zero_predictions_mask = nominator_df["prediction"] == 0
        try:
            zero_predictions = zero_predictions_mask.value_counts().loc[True]
        except KeyError:
            zero_predictions = 0

        total_bond = nominator_df.loc[0, "total_bond"]
        if total_bond == 52704620144862450:
            print()
        difference_to_total_bond = np.subtract(
            total_bond, nominator_df["prediction"].sum()
        )
        mod_difference_to_total_bond = np.mod(
            abs(difference_to_total_bond),
            (len(nominator_df) - zero_predictions),
        )
        if not mod_difference_to_total_bond:
            if int(len(nominator_df) - zero_predictions) == 0:
                print()

            prediction_sum_difference = int(
                np.divide(
                    difference_to_total_bond,
                    int(len(nominator_df) - zero_predictions),
                )
            )
        else:
            if difference_to_total_bond < 0:
                mod_difference_to_total_bond = -mod_difference_to_total_bond
            prediction_sum_difference = int(
                np.divide(
                    (difference_to_total_bond - mod_difference_to_total_bond),
                    (len(nominator_df) - zero_predictions),
                )
            )

            # here we simply add the difference to the first prediction, should not affect the result
            nominator_df.loc[0, "prediction"] = (
                nominator_df.loc[0, "prediction"].astype("int64")
                + mod_difference_to_total_bond
            )

        nominator_df.loc[
            ~zero_predictions_mask, ["prediction"]
        ] = nominator_df.loc[~zero_predictions_mask, "prediction"].add(
            prediction_sum_difference
        )

        sanity_check = np.subtract(
            total_bond, nominator_df["prediction"].sum()
        )
        if sanity_check != 0:
            print(nominator_df)
            print(sanity_check)
            raise ValueError("sanity check failed")

        if (nominator_df["prediction"].values < 0).any():
            print(prediction_sum_difference)
            print(nominator_df)
            raise ValueError("negative stakes")

        return nominator_df

    def preadjustment(self, dataframe):
        """
        This function makes sure that no predictions are negative
        :param dataframe:
        :return: dataframe with no negative predictions
        """

        dataframe.loc[dataframe["prediction"] <= 0, "prediction"] = 1

        return dataframe

    def apply_proportional_split_strategy(self, nominator, dataframe):



        dataframe = self.calculate_difference_expected_sum_stake_and_prediction(dataframe)


        nominator_df = dataframe.loc[
            dataframe["nominator"] == nominator
        ].reset_index(drop=True)
        total_bond = nominator_df['total_bond'].values[0]
        # if the nominator_df consists of only one row, we simply set the prediction equal to the total bond
        if len(nominator_df["prediction"]) == 1:
            nominator_df.loc[0, "prediction"] = total_bond
            return dataframe, nominator_df

        # calculate ratio of prediction to sum of predictions
        nominator_df["ratio"] = (
            nominator_df["prediction"] / nominator_df["prediction"].sum()
        )

        # if the ratio is NaN, we set an even split
        if nominator_df["ratio"].isnull().values.any():
            nominator_df["ratio"] = nominator_df["ratio"].fillna(
                1 / len(nominator_df)
            )

        nominator_df = self.adapt_ratio_to_expected_sum_stake(nominator_df)

        # multiply ratio with total bond
        nominator_df["prediction"] = (
            nominator_df["ratio"] * nominator_df["total_bond"]
        )

        # round prediction to the nearest integer
        nominator_df["prediction"] = np.floor(nominator_df["prediction"])

        # ensure that nominator_df["prediction"] is type int64
        nominator_df["prediction"] = nominator_df["prediction"].astype("int64")

        # calculate difference between total bond and sum of predictions
        difference_to_total_bond = np.subtract(
            total_bond, nominator_df["prediction"].sum()
        )

        # add difference to first prediction
        nominator_df.loc[0, "prediction"] = (
            nominator_df.loc[0, "prediction"] + difference_to_total_bond
        )

        # sanity check
        sanity_check = np.subtract(
            total_bond, nominator_df["prediction"].sum()
        )
        if sanity_check != 0:
            print(nominator_df)
            print(sanity_check)
            raise ValueError("sanity check failed")

        dataframe.loc[dataframe['nominator'] == nominator, 'prediction'] = nominator_df['prediction'].values

        return dataframe, nominator_df

    def calculate_difference_expected_sum_stake_and_prediction_vectorized(self, dataframe):
        """
        difference = dataframe.groupby(["nominator", "validator"])['expected_sum_stake'].nth(0) - \
                     dataframe.groupby(["nominator", "validator"])['prediction'].nth(0)
        difference = np.subtract(difference, difference.min())
        difference = pd.DataFrame(difference).reset_index()
        difference.rename(columns={0: "difference"}, inplace=True)
        dataframe['difference'] = difference['difference'].values
        """

        difference = dataframe.groupby(["validator"])['expected_sum_stake'].mean() - \
                     dataframe.groupby(["validator"])['prediction'].sum()
        difference = np.subtract(difference, difference.min())
        difference = pd.DataFrame(difference).reset_index()
        difference.rename(columns={0: "difference"}, inplace=True)
        dataframe = dataframe.merge(difference, on="validator", how="left")

        return dataframe


    def apply_proportional_split_strategy_vectorized(self, dataframe):
        """
        This function applies the proportional split strategy to the dataframe
        :param dataframe:
        :return: dataframe with predictions
        """

        dataframe = self.calculate_difference_expected_sum_stake_and_prediction_vectorized(dataframe)


        dataframe['ratio'] = dataframe.groupby("nominator")['prediction'].transform(lambda x: x / x.sum())


        # if the ratio is NaN, we set an even split
        if dataframe["ratio"].isnull().values.any():
            dataframe.loc[dataframe['ratio'].isnull()] = dataframe.groupby("nominator")['ratio'].transform(lambda x: x.fillna(1 / len(x)))


        dataframe = self.adapt_ratio_to_expected_sum_stake_vectorized(dataframe)

        # multiply ratio with total bond
        dataframe["prediction"] = (
            dataframe["ratio"] * dataframe["total_bond"]
        )

        # round prediction to the nearest integer
        dataframe["prediction"] = np.floor(dataframe["prediction"])

        # ensure that nominator_df["prediction"] is type int64
        dataframe["prediction"] = dataframe["prediction"].astype("int64")
        dataframe["total_bond"] = dataframe["total_bond"].astype("int64")


        # calculate difference between total bond and sum of predictions
        difference_to_total_bond = pd.DataFrame(dataframe.groupby("nominator")['total_bond'].first() - \
                                   dataframe.groupby("nominator")['prediction'].sum())


        difference_to_total_bond.reset_index(inplace=True)
        difference_to_total_bond.sort_values(by="nominator", inplace=True)

        dataframe.sort_values(by="nominator", inplace=True)

        dataframe.loc[dataframe.groupby("nominator").head(1).index, "prediction"] = dataframe.loc[dataframe.groupby("nominator").head(1).index, "prediction"] + difference_to_total_bond[0].values


        # sanity check
        sanity_check = np.subtract(dataframe.groupby("nominator")['total_bond'].first(),
                                   dataframe.groupby("nominator")['prediction'].sum())
        if sanity_check.sum() != 0:
            print(dataframe)
            print(sanity_check)
            raise ValueError("sanity check failed")

        return dataframe

    def update_prediction_in_dataframe(self, nominator_df, dataframe):
        """
        This function updates the prediction in the dataframe
        :param nominator_df:
        :param dataframe:
        :return:
        """
        dataframe.loc[dataframe['nominator'] == nominator_df['nominator'].values[0], 'prediction'] = nominator_df['prediction'].values[0]
        return dataframe

    def adapt_ratio_to_expected_sum_stake_vectorized(self, dataframe):
        """
        This function adapts the ratio of the prediction to the expected sum stake
        :param nominator_df: dataframe with columns: nominator, validator, proportional_bond, total_bond, number_of_validators, total_proportional_bond, era, solution_bond, prediction
        :return: dataframe with adapted ratio
        """

        # normalize difference
        dataframe['difference'] = dataframe.groupby("nominator")['difference'].transform(lambda x: x / x.sum())

        if dataframe['difference'].isnull().any():
            dataframe['difference'] = dataframe['difference'].fillna(1)

        dataframe["ratio"] =  dataframe["ratio"] + dataframe["difference"]
        #dataframe['ratio'] = dataframe[['ratio', 'difference']].max(axis=1)
        dataframe["ratio"] = dataframe.groupby("nominator")['ratio'].transform(lambda x: x / x.sum())
        return dataframe

    def adapt_ratio_to_expected_sum_stake(self, nominator_df):
        """
        This function adapts the ratio of the prediction to the expected sum stake
        :param nominator_df: dataframe with columns: nominator, validator, proportional_bond, total_bond, number_of_validators, total_proportional_bond, era, solution_bond, prediction
        :return: dataframe with adapted ratio
        """

        # normalize difference

        nominator_df["difference"] = nominator_df["difference"] / nominator_df["difference"].sum()
        #nominator_df["ratio"] =  nominator_df["ratio"] + nominator_df["difference"]
        nominator_df['ratio'] = nominator_df[['ratio', 'difference']].max(axis=1)
        nominator_df["ratio"] = nominator_df["ratio"] / nominator_df["ratio"].sum()
        return nominator_df

    def calculate_difference_expected_sum_stake_and_prediction(self, dataframe):

        difference = dataframe.groupby("validator")['expected_sum_stake'].mean() - dataframe.groupby("validator")['prediction'].sum()
        difference = difference - difference.min()
        dataframe["difference"] = dataframe["validator"].map(difference)
        return dataframe


    def adjust_solver_solution(self,dataframe, total_bond_df):


        # stack the dataframe
        dataframe = dataframe.stack().reset_index()

        # rename columns
        dataframe.columns = ['nominator', 'validator', 'prediction']


        # merge total_bond dataframe with dataframe
        dataframe = total_bond_df.merge(dataframe, on=['nominator', 'validator'], how='left')

        # calculate ratio
        dataframe['ratio'] = dataframe.groupby("nominator")['prediction'].transform(lambda x: x / x.sum())

        # multiply ratio with total bond
        dataframe["prediction"] = (
            dataframe["ratio"] * dataframe["total_bond"]
        )

        # round prediction to the nearest integer
        dataframe["prediction"] = np.floor(dataframe["prediction"])

        # ensure that nominator_df["prediction"] is type int64
        dataframe["prediction"] = dataframe["prediction"].astype("int64")

        # Calculate difference between total bond and sum of predictions
        difference = dataframe.groupby("nominator")['total_bond'].first() - \
                     dataframe.groupby("nominator")['prediction'].sum()

        # map difference to dataframe
        dataframe["difference"] = dataframe["nominator"].map(difference)

        # Calculate the indices of the first occurrence of each nominator
        first_indices = dataframe.groupby("nominator").nth(0).index

        # Add positive differences to the first validator of each nominator


        dataframe.loc[dataframe.index.isin(first_indices), 'prediction'] += dataframe.loc[first_indices, 'difference']

        # Subtract negative differences from the first validator of each nominator
        #dataframe.loc[dataframe.index.isin(first_indices) & negative_mask, 'prediction'] += dataframe.loc[first_indices, 'difference']

        # sanity check
        sanity_check = np.subtract(dataframe.groupby("nominator")['total_bond'].first(),
                                   dataframe.groupby("nominator")['prediction'].sum())
        if sanity_check.sum() != 0:
            print(dataframe)
            print(sanity_check)
            raise ValueError("sanity check failed")

        return dataframe


    def proportional_split_strategy(self, dataframe):
        """
        This function groups by nominator, sums up the prediction values, compares with the total bond (which is the 100% benchmark) and adjusts the prediction values accordingly
        There are two cases:
        1 The difference divides nicely into the number of validators, then the difference is evenly
        distributed among the validators
        2 The difference does not divide nicely into the number of validators, then the difference is
        evenly distributed among the validators and the first validator gets the remainder
        :param dataframe: dataframe with columns: nominator, validator, proportional_bond, total_bond, number_of_validators, total_proportional_bond, era, solution_bond, prediction
        :return: adjusted prediction values in prediction column
        """
        if dataframe is None:
            raise ValueError("dataframe is None")

        dataframe["prediction"] = dataframe["prediction"].astype("int64")
        dataframe.reset_index(drop=True, inplace=True)

        dataframe = self.preadjustment(dataframe)
        return self.apply_proportional_split_strategy_vectorized(dataframe)

        """
        nominators = dataframe["nominator"].unique()
        counter = 0

        # this is the single process version // DEBUGGING only
        start = time.time()
        results = []
        for nominator in nominators:
            counter += 1
            print(f"nominator {counter} of {len(nominators)}")
            dataframe, nominator_df = self.apply_proportional_split_strategy(nominator, dataframe)
            results.append(nominator_df)

        end = time.time()
        print(f"single process version took {end - start} seconds")"""
        """
        # this is the multiprocessing version
        #with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(
                self.apply_proportional_split_strategy,
                [(nominator, dataframe) for nominator in nominators],
            )
        # unpack results
        results = [result[1] for result in results]


        adjusted_dataframe = pd.concat(results)

        return adjusted_dataframe"""


    def add_removed_rows(self):
        """
        add rows that were removed from the training data
        :return: dataframe with added rows
        """
        era = self.dataframe["era"].max()
        path = f"../data_collection/data/model_2/removed_data_{era}.csv"
        removed_dataframe = pd.read_csv(path)
        removed_dataframe = removed_dataframe.loc[removed_dataframe["era"] == era]
        self.dataframe = pd.concat([self.dataframe, removed_dataframe])
        self.dataframe.reset_index(drop=True, inplace=True)

    def insert_predictions_removed_rows(self):
        """
        insert predictions for rows that were removed from the training data. It does this by grouping by nominator and
        summing up the predictions. Then it compares the sum of the predictions with the total bond and sets the missing
        prediction to the difference. Rows with only one validator are set to the total bond.
        :return:
        """

        difference = self.dataframe.groupby("nominator")['total_bond'].mean() - self.dataframe.groupby("nominator")['prediction'].sum()
        self.dataframe.loc[self.dataframe['prediction'].isnull(), 'prediction'] = self.dataframe['nominator'].map(difference)
        return self.dataframe



    def even_split_strategy(self, dataframe=None):
        """
        This function groups by nominator, sums up the prediction values, compares with the total bond (which is the 100% benchmark) and adjusts the prediction values accordingly
        There are two cases:
        1 The difference divides nicely into the number of validators, then the difference is evenly
        distributed among the validators
        2 The difference does not divide nicely into the number of validators, then the difference is
        evenly distributed among the validators and the first validator gets the remainder
        :param dataframe: dataframe with columns: nominator, validator, proportional_bond, total_bond, number_of_validators, total_proportional_bond, era, solution_bond, prediction
        :return: adjusted prediction values in prediction column
        """
        if dataframe is None:
            raise ValueError("dataframe is None")

        dataframe["prediction"] = dataframe["prediction"].astype(int)
        dataframe.reset_index(drop=True, inplace=True)
        nominators = dataframe["nominator"].unique()
        counter = 0

        dataframe = self.preadjustment(dataframe)

        # this is the single process version
        for nominator in nominators:
            self.apply_even_split_strategy(nominator, dataframe)
        counter += 1

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(
                self.apply_even_split_strategy,
                [(nominator, dataframe) for nominator in nominators],
            )
        adjusted_dataframe = pd.concat(results)

        return adjusted_dataframe

    def weighted_split_strategy(self):
        return

    def adjust_top_ends_strategy(self):
        return

    def keep_adjustment_score_substrategy(self):
        return

    @staticmethod
    def adjust_negative_stakes_substrategy_reduce_maxindex(dataframe):
        """
        This function adjusts negative stakes to 0 by finding the indices of the negative stakes and setting them to 0
        and subtracting the absolute value of the negative stake from the maximum prediction value row.
        :param dataframe:
        :return:
        """
        max_index = dataframe["prediction"].idxmax()
        value_to_subtract = (
            dataframe.loc[dataframe["prediction"] < 0, "prediction"]
            .abs()
            .sum()
        )
        dataframe.loc[dataframe["prediction"] < 0, "prediction"] = 0
        dataframe.loc[max_index, "prediction"] = (
            dataframe.loc[max_index, "prediction"] - value_to_subtract
        )
        return dataframe

    def adjust_negative_stakes_substrategy_apply_abs(self, dataframe):
        """
        This function adjusts negative stakes to absolute values
        :param dataframe:
        :return:
        """
        dataframe["prediction"] = dataframe["prediction"].abs()
        return dataframe

    def adjust_cvxpy_strategy(self, dataframe=None):

        if dataframe is None:
            raise ValueError("dataframe is None")

        dataframe["prediction"] = dataframe["prediction"].astype(int)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe = self.preadjustment(dataframe)

        # Define variables
        pred = cp.Variable(len(dataframe))

        # Define constraints
        constraints = []
        for n, g in dataframe.groupby(["nominator"]):
            idx = g.index
            target = g["total_bond"].iloc[0]
            constraints.append(cp.sum(pred[idx]) == target)

        # constraints.append(cp.sum(pred) == cp.sum(data['target']))

        # Define objective
        objective = cp.Minimize(cp.norm(pred - dataframe["prediction"]))

        print(cp.installed_solvers())
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK", verbose=True)

        # Get optimized predictions
        dataframe["optimized_prediction"] = pred.value
        return dataframe


if __name__ == "__main__":
    """
    print("Number of cpu : ", multiprocessing.cpu_count())

    dataframe = pd.read_csv("../../data_collection/data/solved_solutions/947_solved.csv", index_col=0)
    dataframe_index = pd.read_csv("../../data_collection/data/solved_solutions/947_index.csv", index_col=0)
    dataframe_total_bond = pd.read_csv("../../data_collection/data/processed_data/model_3_data_947.csv", index_col=0)
    # apply index to dataframe
    dataframe.columns = dataframe_index.columns
    dataframe = dataframe.set_index(dataframe_index.index)
    """
    # nominator, validator, total_bond, prediction, expected_sum_stake
    example_dataframe = [
        ["nominator_1", "validator_1", 100, 0, 100],
        ["nominator_1", "validator_2", 100, 20, 150],
        ["nominator_1", "validator_3", 100, 50, 50],
        ["nominator_2", "validator_1", 100, 90, 100],
        ["nominator_2", "validator_2", 100, 10, 150],
        ["nominator_2", "validator_3", 100, 0, 50],
        ["nominator_3", "validator_1", 100, 200, 100],
        ["nominator_3", "validator_2", 100, 100, 150],
        ["nominator_3", "validator_3", 100, 50, 50],
    ]
    example_dataframe = pd.DataFrame(example_dataframe,
                                     columns=["nominator", "validator", "total_bond", "prediction",
                                              "expected_sum_stake"])
    adjustment_tool = AdjustmentTool()
    adjusted = adjustment_tool.proportional_split_strategy(example_dataframe)
    print(adjustment_tool.dataframe)