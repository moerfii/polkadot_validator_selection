import json
from src.get_data import StakingSnapshot
import numpy as np
from src.score_data import ScoringUtility

"""
This code is trash, only for benchmarking purposes
"""

path = "../data/snapshot_data/"
with open("./snapshot_data/590_snapshot.json", "rb") as jsonfile:
    data = json.load(jsonfile)

snapshot_instance = StakingSnapshot(config_path="../config.json")
json_winners, json_assignments = snapshot_instance.calculate_optimal_solution(
    path, "10"
)

assignment_dict = {}
for voter in data["voters"]:
    for validator in voter[2]:
        try:
            assignment_dict[validator] += voter[1] / len(voter[2])
        except KeyError:
            assignment_dict[validator] = voter[1] / len(voter[2])


top_boys = np.argpartition(np.array(list(assignment_dict.values())), 297)[
    -297:
]

assigment_dict_list = []
for boy in top_boys:
    temp = (boy, assignment_dict[str(boy)])
    assigment_dict_list.append(temp)


scorer = ScoringUtility()
score1 = scorer.calculate_score(assigment_dict_list)
score2 = scorer.calculate_score(json_winners)
scorer.is_score1_better_than_score2(score1, score2)

compare_boys = set()
for winner in json_winners:
    compare_boys.add(winner[0])
top_boys = np.argpartition(np.array(list(assignment_dict.values())), 297)[
    -297:
]

top_boys_stringed = set()
for top in top_boys:
    top_boys_stringed.add(str(top))


print()
