import numpy as np


"""
ONLY WORKS WITH COMPLETE SINGULAR ERAS
get test_dataframe with predictions made by model
go through nominators, add up total stakes and adjust to 100%
implement various adjustment strategies
"""


class AdjustmentTool:
    def even_split_strategy(self, dataframe):
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
        dataframe["prediction"] = dataframe["prediction"].astype(int)
        dataframe.reset_index(drop=True, inplace=True)
        nominators = dataframe["nominator"].unique()
        counter = 0
        for nominator in nominators:
            counter += 1
            print(counter)
            nominator_df = dataframe.loc[dataframe["nominator"] == nominator]
            while (nominator_df["prediction"].values < 0).any():
                nominator_df = self.adjust_negative_stakes_substrategy(
                    nominator_df
                )
            total_bond = nominator_df["total_bond"].iloc[0]
            difference_to_total_bond = np.subtract(
                total_bond, nominator_df["prediction"].sum()
            )
            mod_difference_to_total_bond = np.mod(
                difference_to_total_bond, len(nominator_df)
            )
            if not mod_difference_to_total_bond:
                prediction_sum_difference = int(
                    np.divide(
                        difference_to_total_bond, int(len(nominator_df))
                    )
                )
            else:
                prediction_sum_difference = int(
                    np.divide(
                        difference_to_total_bond
                        - mod_difference_to_total_bond,
                        len(nominator_df),
                    )
                )
                nominator_df["prediction"].iloc[0] = int(
                    nominator_df["prediction"].iloc[0]
                    + mod_difference_to_total_bond
                )

            nominator_df.loc[:, "prediction"] = nominator_df["prediction"].add(
                prediction_sum_difference
            )
            sanity_check = np.subtract(
                total_bond, nominator_df["prediction"].sum()
            )
            dataframe.loc[dataframe["nominator"] == nominator] = nominator_df
            print(sanity_check)
        return dataframe

    def weighted_split_strategy(self):
        return

    def adjust_top_ends_strategy(self):
        return

    def keep_adjustment_score_substrategy(self):
        return

    @staticmethod
    def adjust_negative_stakes_substrategy(dataframe):
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
        dataframe.loc[max_index, "prediction"] = (
            dataframe.loc[max_index, "prediction"] - value_to_subtract
        )
        dataframe.loc[dataframe["prediction"] < 0, "prediction"] = 0
        return dataframe


if __name__ == "__main__":
    print()
