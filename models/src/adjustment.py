import numpy as np
import multiprocessing

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
        while (nominator_df["prediction"].values < 0).any():
            if len(nominator_df) == 1:
                nominator_df.loc[0, "prediction"] = 0
            else:
                nominator_df = self.adjust_negative_stakes_substrategy_apply_abs(
                    nominator_df
                )
        total_bond = nominator_df.loc[0, "total_bond"]
        difference_to_total_bond = np.subtract(
            total_bond, nominator_df["prediction"].sum()
        )
        mod_difference_to_total_bond = np.mod(
            difference_to_total_bond, len(nominator_df)
        )
        if not mod_difference_to_total_bond:
            prediction_sum_difference = int(
                np.divide(difference_to_total_bond, int(len(nominator_df)))
            )
        else:
            prediction_sum_difference = int(
                np.divide(
                    difference_to_total_bond
                    - mod_difference_to_total_bond,
                    len(nominator_df),
                )
            )
            nominator_df.loc[0, "prediction"] = nominator_df.loc[0, "prediction"].astype("int64") \
                                                        + mod_difference_to_total_bond

        if prediction_sum_difference > 0 or len(nominator_df)==1:
            nominator_df.loc[:, ["prediction"]] = nominator_df.loc[:, "prediction"].add(
                prediction_sum_difference
            )
        else:
            nominator_df.loc[:, ["prediction"]] = nominator_df.loc[:, "prediction"].add(
                prediction_sum_difference
            )
            while (nominator_df["prediction"].values < 0).any():
                nominator_df = self.adjust_negative_stakes_substrategy_reduce_maxindex(
                    nominator_df
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

        """ This is for testing purposes
        for nominator in nominators:
            self.apply_even_split_strategy(nominator, dataframe)
        """
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


if __name__ == "__main__":
    print("Number of cpu : ", multiprocessing.cpu_count())

    df = pd.read_csv("../../data_collection/data/model_2/df_bond_distribution_testing_0.csv")
    adjustment_tool = AdjustmentTool(df)
    adjustment_tool.even_split_strategy()
