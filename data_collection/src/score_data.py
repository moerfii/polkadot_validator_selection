import json
import os

import numpy as np


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

        # Calculate the sum of the stakes
        sum_of_stakes = int(np.sum(stakes_array))

        # Calculate the variance of the stakes
        variance_of_stakes = int(np.var(stakes_array))

        # Calculate the minimum value of the stakes
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
            f"storedmin: {scores2_array[0]}, calcmin: {int(scores1_array[0])}"
        )
        if scores2_array[0] > int(scores1_array[0]):
            return False

        # compare sum stakes: goal is to maximise: if calculated is worse return False
        print(
            f"storedsum: {scores2_array[1]}, calcsum: {int(scores1_array[1])}"
        )
        if scores2_array[1] > int(scores1_array[1]):
            return False

        # compare variance of stakes: goal is to minimise: if calculated is worse return False
        print(
            f"storedvar: {scores2_array[2]}, calcvar: {int(scores1_array[2])}"
        )
        if scores2_array[2] < int(scores1_array[2]):
            return False
        return True

    def check_correctness_solution(self, snapshot_data, calculated_solution):
        # check if all the edges present in calc solution are also present in snapshot, if this is the case return True
        # more of a sanity check
        return True


if __name__ == "__main__":



    scorer = ScoringUtility()
    path = "../data/calculated_solutions_data/948_winners.json"
    score = scorer.calculate_score(scorer.read_json(path))


    print(score)
    breakpoint()

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
        scores2.append(np.asarray(scorer.read_json(path)["raw_solution"]["score"]))

    for index, value in enumerate(scores1):
        print(index)
        print(scorer.is_score1_better_than_score2(value, scores2[index]))
    print()
    """
    
    import pandas as pd

    dataframe = pd.read_csv("../data/model_2/990_max_min_stake.csv")

    with open(
        "../data/stored_solutions_data/990_stored_solution.json", "r"
    ) as jsonfile:
        snapshot_data = json.load(jsonfile)

    # print score of stored solution
    score = snapshot_data["raw_solution"]["score"]
    print(f"score of stored solution: {score}")

    # print mean of dataframe
    # print minimuum stake of dataframe
    print(dataframe.head())
    # drop the first column
    dataframe = dataframe.drop(dataframe.columns[0], axis=1)
    minimum = dataframe.sum(axis=0).min()
    print(f"minimum stake of dataframe: {minimum}")
        """
