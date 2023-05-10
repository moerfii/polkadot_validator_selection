import numpy as np
import json
import os


class ScoringUtility:
    def __init__(self):
        self.calculated_score = None

    @staticmethod
    def read_json(path):
        with open(path, "r") as jsonfile:
            return json.load(jsonfile)

    @staticmethod
    def calculate_score(json_file=None):
        if json_file is None:
            raise UserWarning("Must provide json to calculate score.")
        stakes = []
        for row in json_file:
            stakes.append(row[1])
        stakes_array = np.array(stakes)

        sum_of_stakes = int(np.sum(stakes_array))
        variance_of_stakes = int(np.var(stakes_array))
        min_stake = int(np.min(stakes_array))

        return np.asarray([min_stake, sum_of_stakes, variance_of_stakes])

    @staticmethod
    def is_score1_better_than_score2(scores1=None, scores2=None):
        if scores1 is None or scores2 is None:
            raise UserWarning("Must provide 2 scores lists")

        scores1_array = np.asarray(scores1)
        scores2_array = np.asarray(scores2)

        if not scores2_array[1]:
            print("Bad solution stored")
            return True
        # compare min stake: goal is to maximise: if calculated is worse return False
        print(
            f"stored min: {scores2_array[0]}, predicted min: {int(scores1_array[0])}"
        )
        if scores2_array[0] > int(scores1_array[0]):
            return False

        # compare sum stakes: goal is to maximise: if calculated is worse return False
        print(
            f"stored sum: {scores2_array[1]}, predicted sum: {int(scores1_array[1])}"
        )
        if scores2_array[1] > int(scores1_array[1]):
            return False

        # compare variance of stakes: goal is to minimise: if calculated is worse return False
        print(
            f"stored var: {scores2_array[2]}, predicted var: {int(scores1_array[2])}"
        )
        if scores2_array[2] < int(scores1_array[2]):
            return False
        return True

    def check_correctness_solution(self, snapshot_data, calculated_solution):
        # check if all the edges present in calc solution are also present in snapshot, if this is the case return True
        # more of a sanity check
        return True


class ScoringTool:
    @staticmethod
    def score_solution(solution):
        sum_of_stakes = int(np.sum(solution))
        variance_of_stakes = int(np.var(solution))
        min_stake = int(np.min(solution))

        return np.asarray([min_stake, sum_of_stakes, variance_of_stakes])

    @staticmethod
    def dataframe_groupby_predictions_by_validators(dataframe):
        """
        Groups the predictions by the validator (sums up) and returns a dataframe
        """
        return (
            dataframe.loc[:, ["validator", "prediction"]]
            .groupby("validator")
            .sum()
        )


if __name__ == "__main__":
    scorer = ScoringUtility()
    dirs = sorted(os.listdir("../data/calculated_solutions_data/"))
    scores1 = []
    for dir in dirs:
        if "_winners" in dir:
            path = "../data/calculated_solutions_data/" + dir
            scores1.append(scorer.calculate_score(scorer.read_json(path)))
    dirs1 = sorted(os.listdir("../data/stored_solutions_data/"))
    scores2 = []
    for dir in dirs1:
        print(dir)
        path = "../data/stored_solutions_data/" + dir
        scores2.append(
            np.asarray(scorer.read_json(path)["raw_solution"]["score"])
        )

    for index, value in enumerate(scores1):
        print(index)
        print(scorer.is_score1_better_than_score2(value, scores2[index]))
    print()
