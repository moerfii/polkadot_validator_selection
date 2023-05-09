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


    def apply_even_split_strategy(self,nominator, dataframe):


        nominator_df = dataframe.loc[dataframe["nominator"] == nominator].reset_index(drop=True)
        # if the nominator_df consists of only one row, we simply set the prediction equal to the total bond
        if len(nominator_df['prediction']) == 1:
            nominator_df.loc[0, "prediction"] = nominator_df.loc[0, "total_bond"]
            return nominator_df

        print(nominator_df)
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
            abs(difference_to_total_bond), (len(nominator_df) - zero_predictions))
        if not mod_difference_to_total_bond:
            prediction_sum_difference = int(np.divide(difference_to_total_bond, int(len(nominator_df)-zero_predictions)))
        else:
            if difference_to_total_bond < 0:
                mod_difference_to_total_bond = -mod_difference_to_total_bond
            prediction_sum_difference = int(np.divide(
                    (difference_to_total_bond - mod_difference_to_total_bond),
                    (len(nominator_df) - zero_predictions)))

            # here we simply add the difference to the first prediction, should not affect the result
            nominator_df.loc[0, "prediction"] = nominator_df.loc[0, "prediction"].astype("int64") \
                                                        + mod_difference_to_total_bond

        nominator_df.loc[~zero_predictions_mask, ["prediction"]] = nominator_df.loc[~zero_predictions_mask, "prediction"]\
            .add(prediction_sum_difference
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
        This function makes sure that no predictions are negative and that no predictions are bigger than the value defined in "total_bond".

        :param dataframe:
        :return:
        """

        for index, row in dataframe.iterrows():
            if row["prediction"] < 0:
                dataframe.loc[index, "prediction"] = 0
            if row["prediction"] > row["total_bond"]:
                dataframe.loc[index, "prediction"] = row["total_bond"]
        return dataframe

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
            dataframe = self.dataframe

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
            results = pool.starmap(self.apply_even_split_strategy, [(nominator, dataframe) for nominator in nominators])
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

        # Define variables
        pred = cp.Variable(len(dataframe))

        # Define constraints
        constraints = []
        for n, g in dataframe.groupby(['nominator']):
            idx = g.index
            target = g['total_bond'].iloc[0]
            constraints.append(cp.sum(pred[idx]) == target)

        # constraints.append(cp.sum(pred) == cp.sum(data['target']))

        # Define objective
        objective = cp.Minimize(cp.norm(pred - dataframe['prediction']))

        print(cp.installed_solvers())
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK", verbose=True)

        # Get optimized predictions
        dataframe['optimized_prediction'] = pred.value
        return dataframe

if __name__ == "__main__":
    print("Number of cpu : ", multiprocessing.cpu_count())

    df = pd.read_csv("../../data_collection/data/model_2/df_bond_distribution_testing_0.csv")
    adjustment_tool = AdjustmentTool(df)
    adjustment_tool.even_split_strategy()
