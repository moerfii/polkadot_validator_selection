import json
import os
import pandas as pd
from src.get_data import StakingSnapshot
from src.preprocessor import Preprocessor
from src.utils import read_json, progress_of_loop
import argparse
from pathlib import Path

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


def get_snapshot_data(snapshot_instance):
    return snapshot_instance.get_snapshot()


def get_solution_data(snapshot_instance):
    return snapshot_instance.get_stored_solution()


def save_dataframe(dataframe, path):
    dataframe.to_parquet(path)


def get_data(
    snapshot_instance, block_numbers, save_to_json=False, req_dirs=None
):
    block_numbers.sort_values("Era", inplace=True)
    block_number_counter = 0
    for index, row in block_numbers.iterrows():
        progress_of_loop(block_number_counter, block_numbers, "get_data")

        snapshot_data = None
        nominator_mapping = None
        validator_mapping = None
        solution_data = None
        json_winners = None
        json_assignments = None

        snapshot_path = req_dirs[0]
        snapshot_path_file = snapshot_path + row["Era"] + "_snapshot.json"
        stored_solution_path = req_dirs[1]
        stored_solution_path_file = (
            stored_solution_path + row["Era"] + "_stored_solution.json"
        )
        calculated_solution_path = req_dirs[2]
        calculated_solution_path_file = (
            calculated_solution_path + row["Era"] + "_winners.json"
        )

        # check if the era has already been handled, improves speed of system overall.
        block_number_snapshot = row["SignedPhaseBlock"]
        block_number_solution = row["SolutionStoredBlock"] + 1
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

        if save_to_json and snapshot_data is not None:
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

        if save_to_json and solution_data is not None:
            snapshot_instance.write_to_json(
                name="_stored_solution.json",
                data_to_save=solution_data,
                storage_path=stored_solution_path,
            )

        if not os.path.exists(calculated_solution_path_file):
            # calculate solution with custom phragmen rust script & only save if its equal or better to the stored one.
            snapshot_instance.set_block_number(block_number_snapshot)
            (
                json_winners,
                json_assignments,
            ) = snapshot_instance.calculate_optimal_solution(
                snapshot_path, iterations="400"
            )
            # todo: ScoringUtlity(snapshot_instance.era, json_assignments, snapshot_data).check_correctness_solution()
        if save_to_json and json_winners is not None:
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


def preprocess_data(req_dirs):
    # process snapshots
    snap_path = req_dirs[0]
    snapshots = sorted(os.listdir(snap_path))
    snapshots = [snap for snap in snapshots if "mapping" not in snap]
    solution_path = req_dirs[2]
    solutions = sorted(os.listdir(solution_path))
    solutions = [sol for sol in solutions if "assignments" not in sol]
    snapshots_list = []
    snapshot_counter = 0
    for index, snap in enumerate(snapshots):
        era = [int(s) for s in snap.split("_") if s.isdigit()][0]
        progress_of_loop(
            snapshot_counter, snapshots, "Preprocessing Snapshots"
        )
        with open(snap_path + snap, "r") as snapjson:
            snapshot_json = json.load(snapjson)
        with open(solution_path + solutions[index], "r") as soljson:
            solution_json = json.load(soljson)
        snapshots_list.append(
            Preprocessor.process_snapshot_data(
                snapshot_json, era, snapshots_list, solution_json
            )
        )

    """    solutions_counter = 0
    for index, sol in enumerate(solutions):
        solutions_counter += 1
        bagsprogress = (solutions_counter / len(solutions)) * 100
        sys.stdout.write("Preprocessing Snapshots Progress: %d%%   \r" % bagsprogress)
        sys.stdout.flush()
        with open(solution_path + sol, 'r') as soljson:
            solution_json = json.load(soljson)
        solutions_list.append(Preprocessor.process_solution_data(solution_json,
                                                                 snapshots_list, index))"""
    dataframes = []
    for sub_df in snapshots_list:
        dataframes.append(pd.DataFrame.from_dict(sub_df, orient="index"))
    return pd.concat(dataframes)

    # process calc_solutions


def environment_handler():
    dirs = os.listdir("./data/")
    required_directories = [
        "data/snapshot_data/",
        "data/stored_solutions_data/",
        "data/calculated_solutions_data/",
    ]
    for directory in required_directories:
        strip_dir = Path(directory).parts
        if strip_dir[1] not in dirs:
            os.mkdir(directory)
    return required_directories


def setup():
    required_directories = environment_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--cpath",
        help="Submit the path to the config.json file",
        type=str,
    )
    parser.add_argument(
        "-m", "--mode", help="Select live or historic", type=str
    )
    parser.add_argument(
        "-s", "--save", help="Provide path to save file", type=str
    )
    args = parser.parse_args()
    if args.cpath is None:
        raise UserWarning("Please submit the path to your config.json")
    path = args.cpath
    return StakingSnapshot(), path, required_directories, args


def preprocess_bond_distribution_data(req_dirs):
    """

    :param req_dirs:
    :return:
    """

    snap_path = req_dirs[0]
    snapshots = sorted(os.listdir(snap_path))
    snapshots = [snap for snap in snapshots if "mapping" not in snap]
    solution_path = req_dirs[2]
    solutions = sorted(os.listdir(solution_path))
    solutions = [sol for sol in solutions if "assignments" in sol]

    batch_size = 300

    # todo: account for rounding errors in total_bond/ proportional bond

    """
    get all nominators and their stake
    """
    snapshot_dicts = []
    for snap_file in snapshots[batch_size:]:
        snapshot_dict = {}
        snap, jsonfile = read_json(snap_path + snap_file)
        for nominator in snap["voters"]:
            snapshot_dict[nominator[0]] = []
            snapshot_dict[nominator[0]].append(nominator[1])
            snapshot_dict[nominator[0]].append(nominator[2])
        snapshot_dicts.append(snapshot_dict)
        jsonfile.close()

    """
    get all assignments and their stake
    """
    assignment_dicts = []
    snapshot_counter = 0
    for assignment_file in solutions[batch_size:]:
        progress_of_loop(
            snapshot_counter, solutions, "Preprocessing Assignments"
        )
        assignment_dict = {}
        assignments, jsonfile = read_json(solution_path + assignment_file)
        for assignment in assignments:
            assignment_dict[assignment[0]] = assignment[1]
        assignment_dicts.append(assignment_dict)
        jsonfile.close()

    """
    nominator: [validator, specific_proportional_bond, actual_bond (da vo snap[index] * assignment) 
    """
    final_dictionaries = []
    sub_final_dictionaries = []
    snapshot_counter = 0
    for index, assignments in enumerate(assignment_dicts):
        era = [
            int(s)
            for s in snapshots[batch_size + index].split("_")
            if s.isdigit()
        ][0]
        progress_of_loop(
            snapshot_counter, assignment_dicts, "Preprocessing Assignments"
        )
        final_dictionary = []
        sub_final_dictionary = {}
        snap = snapshot_dicts[index]
        for nominator, assignment in assignments.items():
            bond = snap[nominator][0]
            for validator in assignment:
                proportional_bond = int(bond / len(assignment))
                number_of_validators = len(assignment)
                full_bond = bond
                solution_bond = int((validator[1] / 1e9) * bond)
                final_dictionary.append(
                    [
                        nominator,
                        validator[0],
                        era,
                        proportional_bond,
                        number_of_validators,
                        full_bond,
                        solution_bond,
                    ]
                )
                try:
                    sub_final_dictionary[validator[0]] += proportional_bond
                except KeyError:
                    sub_final_dictionary[validator[0]] = proportional_bond
        final_dictionaries.append(final_dictionary)
        sub_final_dictionaries.append(sub_final_dictionary)

    for index, final in enumerate(final_dictionaries):
        for values in final:
            values.append(sub_final_dictionaries[index][values[1]])

    dataframes = []
    for sub_df in final_dictionaries:
        dataframes.append(pd.DataFrame.from_records(sub_df))
    return pd.concat(dataframes)


if __name__ == "__main__":
    snapshot, path, req_dirs, args = setup()

    """
    #snapshot.create_substrate_connection(path)
    ## MODEL 1 DATA
    block_numbers = read_parquet("./block_numbers/new_block_numbers_dataframe.parquet")
    #get_data(snapshot, block_numbers, True, req_dirs)
    if args.mode == "history":
        print('history')
        get_data(snapshot, block_numbers, True, req_dirs)
    else:
        print('subscribe')

    df = preprocess_data(req_dirs)

    df.rename(columns={0: "total_bond",
                       1: "proportional_bond",
                       2: "list_of_nominators",
                       3: "nominator_count",
                       4: "elected_current_era",
                       5: "elected_previous_era",
                       6: "elected_counter",
                       7: "self_stake",
                       8: "avg_stake_per_nominator",
                       9: "era"},
              inplace=True)

    #if args.save is not None:
    df.to_csv('snapshot_with_solution.csv')"""
    ## MODEL2 DATA
    # get predicted active set (for now winners json?)
    # go through nominators, (open snapshot + assignments json) drop all non-active set validators
    # redistribute proportional stake
    # attach actual bond by going into assignments json
    df = preprocess_bond_distribution_data(req_dirs)
    df.rename(
        columns={
            0: "nominator",
            1: "validator",
            2: "era",
            3: "proportional_bond",
            4: "number_of_validators",
            5: "total_bond",
            6: "solution_bond",
            7: "total_proportional_bond",
        },
        inplace=True,
    )
    df.to_csv("df_bond_distribution_2.csv")
    print("done!")
