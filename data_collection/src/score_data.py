import json
import os

import numpy as np


class ScoringUtlity:

    def __init__(self, era):
        self.era = era
        self.winners_json = self.read_json("../calculated_solutions/", "_winners.json")
        self.stored_json = self.read_json("../storedsolutions/", "_0_storedsolution_.json")
        self.calculated_score = []

    def read_json(self,path_to_json, name):
        full_path = path_to_json + str(self.era) + name
        with open(full_path, 'r') as jsonfile:
            return json.load(jsonfile)

    def calculate_score(self):

        stakes = []
        for row in self.winners_json:
            stakes.append(row[1])
        stakes_array = np.array(stakes)

        # Calculate the sum of the stakes
        sum_of_stakes = np.sum(stakes_array)

        # Calculate the variance of the stakes
        variance_of_stakes = np.var(stakes_array)

        # Calculate the minimum value of the stakes
        min_stake = np.min(stakes_array)
        self.calculated_score.append(min_stake)
        self.calculated_score.append(sum_of_stakes)
        self.calculated_score.append(variance_of_stakes)
        return min_stake, sum_of_stakes, variance_of_stakes

    def compare_scores(self):
        stored_score = np.array(self.stored_json['raw_solution']["score"])

        # compare min stake: goal is to maximise: if calculated is worse return False
        print(f"storedmin: {stored_score[0]}, calcmin: {int(self.calculated_score[0])}")
        if stored_score[0] > int(self.calculated_score[0]):
            return False

        # compare sum stakes: goal is to maximise: if calculated is worse return False
        print(f"storedsum: {stored_score[1]}, calcsum: {int(self.calculated_score[1])}")
        if stored_score[1] > int(self.calculated_score[1]):
            return False

        # compare variance of stakes: goal is to minimise: if calculated is worse return False
        print(f"storedvar: {stored_score[2]}, calcvar: {int(self.calculated_score[2])}")
        if stored_score[2] < int(self.calculated_score[2]):
            return False

        return True


if __name__ == "__main__":
    scorer = ScoringUtlity(590)
    mini, summa, var = scorer.calculate_score()
    print(scorer.compare_scores())

    print()
