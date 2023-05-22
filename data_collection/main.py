import json
import os
import numpy as np
import pandas as pd
from src.get_data import NodeQuery
from src.preprocessor import Preprocessor
from src.utils import (
    read_json,
    progress_of_loop,
    read_parquet,
    partition_into_batches,
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
]


def get_snapshot_data(snapshot_instance):
    return snapshot_instance.get_snapshot()


def get_solution_data(snapshot_instance):
    return snapshot_instance.get_stored_solution()


def save_dataframe(dataframe, path):
    dataframe.to_parquet(path)


    # process calc_solutions




def impute_data(df):

    onehot_columns = [
        "0_x",
        "1_x",
        "2_x",
        "3_x",
        "4_x",
        "5_x",
        "6_x",
        "7_x",
        "8_x",
        "9_x",
        "10_x",
        "11_x",
        "12_x",
        "13_x",
        "14_x",
    ]

    impute_columns = ["proportional_bond_y", "total_proportional_bond_y"]

    for impute_column in impute_columns:
        for column in onehot_columns:
            median = df.loc[df[column] == 1][impute_column].median()
            df.loc[df[column] == 1, impute_column] = df.loc[
                df[column] == 1, impute_column
            ].replace(np.nan, median)

        median = df.loc[
            (df[onehot_columns] == 0).all(axis=1), impute_column
        ].median()
        df.loc[(df[onehot_columns] == 0).all(axis=1), impute_column] = df.loc[
            (df[onehot_columns] == 0).all(axis=1), impute_column
        ].replace(np.nan, median)

    return df


def finish_preprocessing(df, eras):
    # in a next step we subtract the proportional_bond from the previous era from the current era
    # this is done to get the change in the data
    # apply to new dataframe
    subtracted_dataframe = df.copy()
    for era in eras:
        if era == eras[0]:
            continue
        else:
            merged_dataframe = df.loc[df["era"] == era].merge(
                df.loc[subtracted_dataframe["era"] == era - 1],
                on=["nominator", "validator"],
                how="left",
            )
            merged_dataframe = impute_data(merged_dataframe)
            subtracted_dataframe.loc[
                subtracted_dataframe["era"] == era, "proportional_bond"
            ] = (
                merged_dataframe["proportional_bond_x"]
                - merged_dataframe["proportional_bond_y"]
            )
            subtracted_dataframe.loc[
                subtracted_dataframe["era"] == era, "total_proportional_bond"
            ] = (
                merged_dataframe["total_proportional_bond_x"]
                - merged_dataframe["total_proportional_bond_y"]
            )

    # drop all rows where era = eras[0]
    return subtracted_dataframe.loc[subtracted_dataframe["era"] != eras[0]]


def get_model_1_data(args):
    block_numbers = read_parquet(args.block_numbers)
    # get_data(snapshot, block_numbers, True, req_dirs)
    block_numbers.sort_values("Era", inplace=True)
    snapshot_instance = NodeQuery()
    snapshot_instance.set_config_path(args.config_path)
    snapshot_instance.create_substrate_connection()
    block_number_counter = 0
    for index, row in block_numbers.iterrows():
        block_number_counter = progress_of_loop(
            block_number_counter, block_numbers, "get_data"
        )
        snapshot_data = None
        nominator_mapping = None
        validator_mapping = None
        solution_data = None
        json_winners = None
        json_assignments = None

        snapshot_path = required_directories[0]
        snapshot_path_file = snapshot_path + row["Era"] + "_snapshot.json"
        stored_solution_path = required_directories[1]
        stored_solution_path_file = (
            stored_solution_path + row["Era"] + "_stored_solution.json"
        )
        calculated_solution_path = required_directories[2]
        calculated_solution_path_file = (
            calculated_solution_path + row["Era"] + "_winners.json"
        )

        # check if the era has already been handled, improves speed of system overall.
        block_number_snapshot = row["SignedPhaseBlock"]
        block_number_solution = row["SolutionStoredBlock"] + 1
        try:
            if not os.path.exists(snapshot_path_file):
                # acquire snapshot
                snapshot_instance.set_block_number(block_number_snapshot)
                snapshot_data = get_snapshot_data(snapshot_instance)
                (
                    nominator_mapping,
                    validator_mapping,
                ) = Preprocessor().return_mapping_from_address_to_index(
                    snapshot_data
                )

            if args.save and snapshot_data is not None:
                snapshot_instance.write_to_json(
                    name="_snapshot.json",
                    data_to_save=snapshot_data,
                    storage_path=snapshot_path,
                )
                snapshot_instance.write_to_json(
                    name="_snapshot_nominator_mapping.json",
                    data_to_save=nominator_mapping,
                    storage_path=snapshot_path,
                )
                snapshot_instance.write_to_json(
                    name="_snapshot_validator_mapping.json",
                    data_to_save=validator_mapping,
                    storage_path=snapshot_path,
                )

            if not os.path.exists(stored_solution_path_file):
                # acquire stored solution
                snapshot_instance.set_block_number(block_number_solution)
                solution_data = get_solution_data(snapshot_instance)

            if args.save and solution_data is not None:
                snapshot_instance.write_to_json(
                    name="_stored_solution.json",
                    data_to_save=solution_data,
                    storage_path=stored_solution_path,
                )

            if not os.path.exists(calculated_solution_path_file):
                print(
                    "calculating solution for era: ",
                    row["Era"],
                    " at block: ",
                    block_number_snapshot,
                    "...",
                )
                # calculate solution with custom phragmen rust script & only save if its equal or better to the stored one.
                snapshot_instance.set_block_number(block_number_snapshot)
                (
                    json_winners,
                    json_assignments,
                ) = snapshot_instance.calculate_optimal_solution(
                    snapshot_path, iterations="400"
                )
                # todo: ScoringUtlity(snapshot_instance.era, json_assignments, snapshot_data).check_correctness_solution()
            if args.save and json_winners is not None:
                snapshot_instance.write_to_json(
                    name="_winners.json",
                    data_to_save=json_winners,
                    storage_path=calculated_solution_path,
                )
                snapshot_instance.write_to_json(
                    name="_assignments.json",
                    data_to_save=json_assignments,
                    storage_path=calculated_solution_path,
                )

        except WebSocketConnectionClosedException:
            print("Connection to node closed, trying to reconnect...")
            main(args)


def process_model_1_data(args):
    preprocessor = Preprocessor()
    df = preprocessor.preprocess_active_set_data()
    df.rename(
        columns={
            0: "total_bond",
            1: "proportional_bond",
            2: "list_of_nominators",
            3: "nominator_count",
            4: "elected_current_era",
            5: "elected_previous_era",
            6: "elected_counter",
            7: "self_stake",
            8: "avg_stake_per_nominator",
            9: "era",
        },
        inplace=True,
    )

    # if args.save is not None:
    df.to_csv("snapshot_with_solution.csv")


def process_model_2_data(args):
        final_dictionaries = []
        sub_final_dictionaries = []
        preprocessor = Preprocessor()
        progress_counter = 0
        if len(args.eras) == 1:
            eras = args.eras
        else:
            eras = range(args.eras[0], args.eras[1])
        for era in eras:
            progress_counter = progress_of_loop(
                progress_counter, eras, "Preprocessing Assignments"
            )
            preprocessor.load_data(era)
            preprocessor.preprocess_model_2_data()
        preprocessor.concatenate_dataframes()
        preprocessor.save_dataframe()
            # load data


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
    #get_model_1_data(args)
    #process_model_1_data(args)
    process_model_2_data(args)
    #get_ensemble_model_2_data(args)


if __name__ == "__main__":
    parser = setup()
    main(parser.parse_args())
