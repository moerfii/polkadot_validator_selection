import os

import pandas as pd

from src.get_data import StakingSnapshot
#from src.preprocessor import Preprocessor
import argparse


"""
step1: acquire data
step1.1: get_snapshot data & stored solutions
read prepared parquet dataframe, storage queries (snapshot&solution) at block numbers, save as json
step 1.2: calculate solutions (phragmen)
read stored solutions, feed into rust script, ensure solution is better & correct, save as json
step2: transform data
tbd
step3: save in persistent manner
tbd
step4: subscribe to block headers for pipeline automation (2 steps: for training and for prediction on the go)
for prediction on the go:
subscribe to events (signedphase started), snapshot query, feed into model, pull posted solution, compare results, post if better
for training:
take snapshot & solution used for prediction on the go, feed into rust script, apply to model.
"""


def get_snapshot_data(snapshot_instance):
    return snapshot_instance.get_snapshot()


def get_solution_data(snapshot_instance):
    return snapshot_instance.get_stored_solution()

def get_calculated_solution_data(snapshot_instance):
    return


def save_dataframe():
    return


def get_data(snapshot_instance, block_numbers):
    # acquire snapshot
    block_number_snapshot = block_numbers['snapshot']
    block_number_solution = block_numbers['solution']
    snapshot_instance.set_block_number(block_number_snapshot)
    snapshot_data = get_snapshot_data(snapshot_instance)
    snapshot_instance.write_to_json(name="_snapshot.json",
                                    data_to_save=snapshot_data,
                                    storage_path="./snapshot_data/")
    # acquire stored solution
    snapshot_instance.set_block_number(block_number_solution)
    solution_data = get_solution_data(snapshot_instance)
    snapshot_instance.write_to_json(name="_stored_solution.json",
                                    data_to_save=solution_data,
                                    storage_path="./stored_solutions_data/")
    # calculate solution with custom phragmen rust script
    snapshot_instance.set_block_number(block_number_snapshot)
    json_winners, json_assignments = snapshot_instance.calculate_optimal_solution("./snapshot_data/", iterations="10")
    snapshot_instance.write_to_json(name="_winners.json",
                                    data_to_save=json_winners,
                                    storage_path="./calculated_solutions_data")
    snapshot_instance.write_to_json(name="_assignments.json",
                                    data_to_save=json_assignments,
                                    storage_path="./calculated_solutions_data")

    return


def environment_handler():
    dirs = os.listdir()
    required_directories = ["stored_solutions_data","snapshot_data","calculated_solutions_data"]
    for dir in required_directories:
        if dir not in dirs:
            os.mkdir(dir)



def setup():
    environment_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--cpath", help="Submit the path to the config.json file", type=str)
    args = parser.parse_args()
    if args.cpath is None:
        raise UserWarning("Please submit the path to your config.json")
    path = args.cpath
    return StakingSnapshot(config_path=path)


def load_parquet(path_to_parquet):
    return pd.read_parquet(path_to_parquet)


if __name__ == "__main__":
    snapshot = setup()
    block_numbers = load_parquet("./block_numbers/block_numbers_dataframe.parquet")
    get_data(snapshot, block_numbers)

