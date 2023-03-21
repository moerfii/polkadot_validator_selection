
import numpy as np

class ScoringTool:

    @staticmethod
    def score_solution(solution):
        # Calculate the sum of the stakes
        sum_of_stakes = int(np.sum(solution))

        # Calculate the variance of the stakes
        variance_of_stakes = int(np.var(solution))

        # Calculate the minimum value of the stakes
        min_stake = int(np.min(solution))

        return np.asarray([min_stake, sum_of_stakes, variance_of_stakes])


    @staticmethod
    def dataframe_groupby_predictions_by_validators(dataframe):
        """
        Groups the predictions by the validator (sums up) and returns a dataframe
        """
        return dataframe.loc[:,['validator', 'prediction']].groupby('validator').sum()
