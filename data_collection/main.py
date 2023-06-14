import json
import os

import numpy as np
import pandas as pd
from .src.get_data import NodeQuery
from .src.get_snapshot import get_snapshot
from .src.preprocessor import Preprocessor
from .src.utils import (
    read_json,
    progress_of_loop,
    read_parquet,
    partition_into_batches,
    set_era_range
)
import argparse
from pathlib import Path
from websocket._exceptions import WebSocketConnectionClosedException

from sklearn.preprocessing import OneHotEncoder


"""
step1: acquire data
step1.1: get_snapshot data & stored solutions
read prepared parquet dataframe, storage queries (snapshot&solution) at block numbers, save as json
step 1.2: calculate solutions (phragmen)
read stored solutions, feed into rust script, ensure solution is better & correct, save as json
step2: transform data
place all in Dataframe with columns (ValidatorAddress, TotalBond, ProportionalBond, *[Nominators], NominatorCount, 
                                        ElectedCurrentEra, ElectedCounter, selfStake, avgStakePerNominator,)
step3: save in persistent manner
save dataframe as parquet
step4: subscribe to block headers for pipeline automation (2 steps: for training and for prediction on the go)
for prediction on the go:
subscribe to events (signedphase started), snapshot query, feed into model, pull posted solution, compare results, post if better
for training:
take snapshot & solution used for prediction on the go, feed into rust script, apply to model.
"""

# global variables
required_directories = [
    "data/snapshot_data/",
    "data/stored_solutions_data/",
    "data/calculated_solutions_data/",
    "data/processed_data/",
]



def get_model_1_data(era):

    snapshot_path = "./data_collection/data/snapshot_data/"

    # get snapshot
    snapshot = get_snapshot(era)

    # get indices
    nominator_mapping, validator_mapping = Preprocessor().return_mapping_from_address_to_index(snapshot)

    # save indices
    with open(f"{snapshot_path}{era}_nominator_mapping.json", "w") as f:
        json.dump(nominator_mapping, f)
    with open(f"{snapshot_path}{era}_validator_mapping.json", "w") as f:
        json.dump(validator_mapping, f)

    # calculate solution (phragmen)
    json_winners, json_assignments = Preprocessor.calculate_optimal_solution(era, snapshot_path, iterations="400")


    # save solution
    calculated_solution_path = "./data_collection/data/calculated_solutions_data/"
    with open(f"{calculated_solution_path}{era}_winners.json", "w") as f:
        json.dump(json_winners, f)
    with open(f"{calculated_solution_path}{era}_assignments.json", "w") as f:
        json.dump(json_assignments, f)



def process_model_1_data(args):
    preprocessor = Preprocessor()
    progress_counter = 0
    eras = set_era_range(args.era)
    for era in eras:
        progress_counter = progress_of_loop(
            progress_counter, eras, "Preprocessing Model 1 Data"
        )
        preprocessor.load_data(era)
        preprocessor.preprocess_model_1_data()
        preprocessor.save_dataframe(args.model_1_path)



def process_model_2_data(args):
        preprocessor = Preprocessor()
        progress_counter = 0
        eras = set_era_range(args.era)
        for era in eras:
            progress_counter = progress_of_loop(
                progress_counter, eras, "Preprocessing Model 2 Data"
            )
            preprocessor.load_data(era)
            preprocessor.preprocess_model_2_data()
            preprocessor.save_dataframe(args.model_2_path)
            preprocessor.group_bonds_by_validator()
            preprocessor.save_dataframe(args.model_2_path + "_grouped")
            preprocessor.preprocess_model_2_Xtest()
            preprocessor.save_dataframe(args.model_2_path + "_Xtest")
            preprocessor.group_bonds_by_validator_Xtest()
            preprocessor.save_dataframe(args.model_2_path + "_grouped_Xtest")



def process_model_3_data(args):
    preprocessor = Preprocessor()
    progress_counter = 0
    eras = set_era_range(args.era)
    for era in eras:
        progress_counter = progress_of_loop(
            progress_counter, eras, "Preprocessing Model 3 Data"
        )
        preprocessor.load_data(era)
        preprocessor.preprocess_model_3_data()
        preprocessor.save_dataframe(args.model_3_path)
        preprocessor.preprocess_model_3_data_Xtest()
        preprocessor.save_dataframe(args.model_3_path + "_Xtest")


def get_ensemble_model_2_data(eras):
    """
    Load model 2 data, iterate over eras and update the target column with the solution proposed by the solver
    :return:
    """
    dataframes = []
    for era in eras:
        print(era)
        # load data
        dataframe = pd.read_csv(f"./data/model_2/{era}_voter_preferences.csv")
        # set index to first column
        dataframe = dataframe.set_index(dataframe.columns[0])
        dataframe = dataframe.stack().reset_index()
        dataframe.rename(
            columns={
                "Unnamed: 0": "nominator",
                "level_1": "validator",
                0: "proportional_bond",
            },
            inplace=True,
        )
        # remove rows that have a value below 0.5
        solution_dataframe = pd.read_csv(
            f"./data/model_2/{era}_max_min_stake.csv"
        )
        solution_dataframe = solution_dataframe.drop(columns=["Unnamed: 0"])
        solution_dataframe = solution_dataframe.stack().reset_index()
        solution_dataframe.rename(columns={0: "solution_bond"}, inplace=True)
        total_dataframe = pd.concat([dataframe, solution_dataframe], axis=1)
        total_dataframe = total_dataframe.drop(columns=["level_0", "level_1"])
        # remove rows that have a value below 0.5
        total_dataframe = total_dataframe[
            total_dataframe["proportional_bond"] > 0.5
        ]

        # group by nominator and get the count of rows to add to new column "nominator_count"
        total_dataframe["nominator_count"] = total_dataframe.groupby(
            "nominator"
        )["nominator"].transform("count")

        # add column era
        total_dataframe["era"] = era

        # save dataframe to csv
        total_dataframe.to_csv(
            f"./data/ensemble_model/{era}_voter_preferences.csv"
        )
        dataframes.append(total_dataframe)
    # concat all dataframes
    total_dataframe = pd.concat(dataframes)
    # save dataframe to csv
    total_dataframe.to_csv(
        f"./data/ensemble_model/total_voter_preferences.csv"
    )


def environment_handler():
    base_dirs = os.listdir("./")
    if "data" not in base_dirs:
        os.mkdir("./data/")
    dirs = os.listdir("./data/")
    for directory in required_directories:
        strip_dir = Path(directory).parts
        if strip_dir[1] not in dirs:
            os.mkdir(directory)
    return required_directories


def setup():
    environment_handler()
    config = "config.json"
    with open(config, "r") as jsonfile:
        config = json.load(jsonfile)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    parser.add_argument(
        "-m", "--mode", help="Select live or historic", type=str
    )
    parser.add_argument(
        "-s", "--save", help="Provide path to save file", type=str
    )
    parser.add_argument("-b", "--batch", help="Provide batch size", type=int)
    parser.add_argument(
        "-e",
        "--eras",
        nargs="+",
        help="range of eras to process, e.g. [500, 600]",
    )
    parser.add_argument(
        "-bn",
        "--block_numbers",
        help="Provide path to block numbers",
        type=str,
    )
    return parser


def main(args):
    get_model_1_data(args)
    process_model_1_data(args)
    #process_model_2_data(args)
    #get_ensemble_model_2_data(args)


if __name__ == "__main__":
    parser = setup()
    main(parser.parse_args())




