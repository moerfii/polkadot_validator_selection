import json
import sys

from src.get_data import StakingSnapshot
#from src.preprocessor import Preprocessor


def get_snapshot_data(snapshot_instance, block_number):
    return


def get_solution_data(snapshot_instance, block_number):
    return


def save_dataframe():
    return


def get_data(snapshot_instance, block_number):
    # get snapshot
    # get solution
    # combine to dataframe, apply any preprocessing
    # save dataframe as parquet
    return


def setup(path_to_config):
    return StakingSnapshot(config_path=path_to_config)


if __name__ == "__main__":
    path = "config.json"
    snapshot = setup(path)
    snapshot.set_block_number(13954650)
    snapshot_stored = snapshot.get_snapshot()
    snapshot.set_block_number(13954656)
    solution_stored = snapshot.get_stored_solution()

