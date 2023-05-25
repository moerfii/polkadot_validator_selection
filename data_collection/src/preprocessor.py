import os
import time

import numpy as np
import pandas as pd
import json

from src.utils import progress_of_loop


class Preprocessor:
    def __init__(self):
        self.snapshot_data = None
        self.winners_data = None
        self.previous_winners_data = None
        self.assignments_data = None
        self.stored_solutions_data = None
        self.era = None
        self.dataframe = None
        self.dataframes = []
        self.removed_dataframe = None
        self.removed_dataframes = []
        self.required_directories = [
            "./data/snapshot_data/",
            "./data/stored_solutions_data/",
            "./data/calculated_solutions_data/",
        ]
        self.output_path = "./data/model_2/"

    def load_data(self, era):
        self.era = era
        try:
            with open(self.required_directories[0] + str(era) + "_snapshot.json") as json_file:
                self.snapshot_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Snapshot data for era {era} not found")
        try:
            with open(self.required_directories[1] + str(era) + "_stored_solution.json") as json_file:
                self.stored_solutions_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Stored solution data for era {era} not found")
        try:
            with open(self.required_directories[2] + str(era) + "_winners.json") as json_file:
                self.winners_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Winners data for era {era} not found")
        try:
            with open(self.required_directories[2] + str(era - 1) + "_winners.json") as json_file:
                self.previous_winners_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Previous winners data for era {era} not found")
        try:
            with open(self.required_directories[2] + str(era) + "_assignments.json") as json_file:
                self.assignments_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Assignments data for era {era} not found")

    def load_snapshot_data(self, snapshot):
        self.voters_dataframes = []
        self.targets_dataframes = []
        for index, file in enumerate(snapshot):
            with open(file) as json_file:
                data = json.load(json_file)
                voters = pd.DataFrame(data["voters"])
                voters.columns = ["Voter", "Weight", "Targets"]
                self.voters_dataframes.append(voters)
                targets = pd.DataFrame(data["targets"])
                targets.columns = ["Targets"]
                self.targets_dataframes.append(targets)

    def load_solution_data(self):
        self.full_targets_dataframes = []
        for index, file in enumerate(self.solutions):
            with open(file) as json_file:
                data = json.load(json_file)
                self.process_solution_data(data)

    @staticmethod
    def process_snapshot_data(
            snapshot, era, previous_processed_dicts, solution
    ):
        # todo: potentially add commission // era points // elected ratio instead of counter
        """
        place all in Dataframe with columns (ValidatorAddress, TotalBond, ProportionalBond, *[Nominators], NominatorCount,
                                        ElectedCurrentEra, ElectedCounter, selfStake, avgStakePerNominator,)
        :return: Dataframe
        """
        voters = snapshot["voters"]
        # targets = snapshot['targets']
        processed_dict = {}
        for row in voters:
            for validator in row[2]:
                try:
                    current_values = processed_dict[validator]

                    total_bond = current_values[0] + row[1]
                    proportional_bond = current_values[1] + row[1] / len(
                        row[2]
                    )
                    list_of_nominators = current_values[2]
                    list_of_nominators.add(row[0])
                    nominator_count = current_values[3] + 1
                    elected_current_era = 0  # get with solution

                    elected_previous_era = current_values[4]
                    elected_counter = current_values[5]
                    self_stake = 0  # potentially add later/should not give extra information
                    avg_stake_per_nominator = proportional_bond / len(
                        list_of_nominators
                    )  # todo: adapt metric.
                    era = era
                    processed_dict[validator] = [
                        total_bond,
                        proportional_bond,
                        list_of_nominators,
                        nominator_count,
                        elected_current_era,
                        elected_previous_era,
                        elected_counter,
                        self_stake,
                        avg_stake_per_nominator,
                        era,
                    ]
                except KeyError:
                    total_bond = row[1]
                    proportional_bond = row[1] / len(row[2])
                    list_of_nominators = set()
                    list_of_nominators.add(row[0])
                    nominator_count = 1
                    elected_current_era = 0
                    elected_previous_era = 0  # check previous
                    elected_counter = 0  # check previous
                    if len(previous_processed_dicts):
                        for previous_dict in reversed(
                                previous_processed_dicts
                        ):
                            try:
                                elected_previous_era = previous_dict[
                                    validator
                                ][
                                    4
                                ]  # es isch ofc da immer 0 will de scheiss ersch nacher processed wird
                                elected_counter = previous_dict[validator][6]
                                break
                            except KeyError:
                                continue
                    else:
                        pass
                    self_stake = 0
                    avg_stake_per_nominator = row[1]
                    era = era
                    processed_dict[validator] = [
                        total_bond,
                        proportional_bond,
                        list_of_nominators,
                        nominator_count,
                        elected_current_era,
                        elected_previous_era,
                        elected_counter,
                        self_stake,
                        avg_stake_per_nominator,
                        era,
                    ]

        return Preprocessor.process_solution_data(
            solution, processed_dict, previous_processed_dicts
        )

    def change_target_values(self, processed_snapshot, solver_solution, era):
        """
        This function changes the target values in the processed_snapshot according to the solver_solution.
        :param processed_snapshot:
        :param solver_solution:
        :return:
        """

        # convert matrix to series
        solver_solution = solver_solution.stack()
        solver_solution = solver_solution[solver_solution >= 1]

        return processed_snapshot

    def preprocess_active_set_data(self):

        # process snapshots
        snap_path = self.required_directories[0]
        snapshots = sorted(os.listdir(snap_path))
        snapshots = [snap for snap in snapshots if "mapping" not in snap]
        solution_path = self.required_directories[2]
        solutions = sorted(os.listdir(solution_path))
        solutions = [sol for sol in solutions if "assignments" not in sol]
        snapshots_list = []
        snapshot_counter = 0
        for index, snap in enumerate(snapshots):
            era = [int(s) for s in snap.split("_") if s.isdigit()][0]
            snapshot_counter = progress_of_loop(
                snapshot_counter, snapshots, "Preprocessing Snapshots"
            )
            with open(snap_path + snap, "r") as snapjson:
                snapshot_json = json.load(snapjson)
            with open(solution_path + solutions[index], "r") as soljson:
                solution_json = json.load(soljson)
            snapshots_list.append(
                self.process_snapshot_data(
                    snapshot_json, era, snapshots_list, solution_json
                )
            )
        dataframes = []
        for sub_df in snapshots_list:
            dataframes.append(pd.DataFrame.from_dict(sub_df, orient="index"))
        return pd.concat(dataframes)

    @staticmethod
    def process_solution_data(
            solution, processed_snapshot_dict, previous_processed_dicts
    ):
        # update processed_dict from snapshot data by checking whether validator got elected and increase counter if so.
        # [total_bond, proportional_bond, list_of_nominators,
        #  nominator_count, elected_current_era, elected_previous_era,
        #  elected_counter, self_stake, avg_stake_per_nominator, era]
        for row in solution:
            processed_snapshot_dict[row[0]][4] = 1
            for processed_dict in reversed(previous_processed_dicts):
                try:
                    processed_snapshot_dict[row[0]][6] = (
                            processed_dict[row[0]][6] + 1
                    )
                    processed_snapshot_dict[row[0]][5] = processed_dict[
                        row[0]
                    ][4]
                    break
                except KeyError:
                    continue
        return processed_snapshot_dict

    @staticmethod
    def return_mapping_from_address_to_index(snapshot):
        validator_mapping_dict = {}
        for index, row in enumerate(snapshot["targets"]):
            validator_mapping_dict[row] = str(index)
        nominator_mapping_dict = {}
        fail_counter = 0
        for index, row in enumerate(snapshot["voters"]):
            nominator_mapping_dict[row[0]] = str(index)
            validator_indices = []
            for validator_address in snapshot["voters"][index][2]:
                try:
                    validator_indices.append(
                        validator_mapping_dict[validator_address]
                    )
                except KeyError:
                    fail_counter += 1
                    # print(f"validator_address: {validator_address} is not in available targets")
        return nominator_mapping_dict, validator_mapping_dict

    @staticmethod
    def map_address_to_index(
            snapshot,
    ):  # todo: refactor to work with mappings saved
        validator_mapping_dict = {}
        for index, row in enumerate(snapshot["targets"]):
            validator_mapping_dict[row] = str(index)
            snapshot["targets"][index] = str(index)
        nominator_mapping_dict = {}
        fail_counter = 0
        for index, row in enumerate(snapshot["voters"]):
            nominator_mapping_dict[row[0]] = str(index)
            validator_indices = []
            for validator_address in snapshot["voters"][index][2]:
                try:
                    validator_indices.append(
                        validator_mapping_dict[validator_address]
                    )
                except KeyError:
                    fail_counter += 1
                    print(
                        f"validator_address: {validator_address} is not in available targets"
                    )
            snapshot["voters"][index] = (
                str(index),
                snapshot["voters"][index][1],
                validator_indices,
            )

        return snapshot, nominator_mapping_dict, validator_mapping_dict

    def add_one_hot_encoding(self):
        """
        This function adds the one hot encoding to the dataframe df.
        :param df:
        :return:
        """
        ## one-hot encoding
        enc = OneHotEncoder(drop="first", handle_unknown="error")
        enc_df = pd.DataFrame(
            enc.fit_transform(
                concatenated_dataframe[["number_of_validators"]]
            ).toarray()
        )
        concatenated_dataframe = concatenated_dataframe.join(enc_df)
        concatenated_dataframe = concatenated_dataframe.drop(
            columns=["number_of_validators"]
        )

    def add_column_previous_scores(self):
        """
        This function adds the column previous_scores to the dataframe df.
        :param df:
        :return:
        """
        previous_era_min_stake = np.min([winner[1] for winner in self.previous_winners_data])
        previous_era_sum_stake = np.sum([winner[1] for winner in self.previous_winners_data])
        previous_era_variance_stake = np.var(
            [winner[1] for winner in self.previous_winners_data]
        )

        self.dataframe["prev_min_stake"] = previous_era_min_stake
        self.dataframe["prev_sum_stake"] = previous_era_sum_stake
        self.dataframe["prev_variance_stake"] = previous_era_variance_stake

    def preprocess_model_2_data(self):

        # prepare snapshot and assignment data
        nominator_validator_mapping = {}
        for nominator in self.snapshot_data["voters"]:
            nominator_validator_mapping[nominator[0]] = []
            nominator_validator_mapping[nominator[0]].append(nominator[1])
            nominator_validator_mapping[nominator[0]].append(nominator[2])


        data = []
        for row in self.assignments_data:
            nominator, assignment = row[0], row[1]
            bond = nominator_validator_mapping[nominator][0]
            for validator in assignment:
                number_of_validators = len(assignment)
                proportional_bond = bond / number_of_validators
                full_bond = bond
                solution_bond = (validator[1] / 1e9) * bond
                data.append(
                    [
                        nominator,
                        validator[0],
                        self.era,
                        proportional_bond,
                        full_bond,
                        number_of_validators,
                        solution_bond,
                    ]
                )
        self.dataframe = pd.DataFrame(data)
        self.dataframe.columns = [
            "nominator",
            "validator",
            "era",
            "proportional_bond",
            "total_bond",
            "number_of_validators",
            "solution_bond",
        ]
        self.add_validator_count()
        self.datatype_casting()
        self.add_column_previous_scores()
        self.add_overall_proportional_bond()
        self.add_overall_total_bond()
        self.add_average_proportional_bond()
        self.add_average_total_bond()
        #self.update_solution_bond()
        self.group_bonds_by_validator()
        #self.add_expected_sum_stake()
        #self.remove_rows_leave_one_validator_out()
        self.dataframes.append(self.dataframe)
        #self.removed_dataframes.append(self.removed_dataframe)

    def concatenate_dataframes(self):
        self.dataframe = pd.concat(self.dataframes)
        #self.removed_dataframe = pd.concat(self.removed_dataframes)

    def add_average_proportional_bond(self):
        """
        This function adds the average proportional bond to the dataframe. It is calculated by dividing the total proportional
        bond by the validator_frequency_current_era
        :return:
        """
        self.dataframe["average_proportional_bond"] = self.dataframe["overall_proportional_bond"] / self.dataframe["validator_frequency_current_era"]

    def add_average_total_bond(self):
        """
        This function adds the average total bond to the dataframe. It is calculated by dividing the total_bond by the validator_frequency_current_era
        :return:
        """
        self.dataframe["average_total_bond"] = self.dataframe["overall_total_bond"] / self.dataframe["validator_frequency_current_era"]


    def datatype_casting(self):
        """
        cast data types to reduce memory usage
        :return:
        """
        dtypes_dict = {
            "era": "int16",
            "total_bond": "int64",
            "proportional_bond": "int64",
            "number_of_validators": "int16",
            "solution_bond": "int64",
        }

        dataframe = self.dataframe.astype(dtype=dtypes_dict)
        return dataframe

    def group_bonds_by_validator(self):
        """
        This function groups the bonds by validator and sums them up. Add nominator and validator columns
        :return:
        """

        self.dataframe = self.dataframe.groupby(["validator"])[['proportional_bond', 'total_bond', 'validator_frequency_current_era', 'overall_proportional_bond', 'solution_bond']].sum().reset_index()
        self.dataframe['era'] = self.era
    def remove_rows_leave_one_validator_out(self):
        """
        This function groups the dataframe by nominator and removes the row with the validator with highest validator count.
        :return:
        """

        self.removed_dataframe = self.dataframe.loc[self.dataframe.groupby("nominator")["overall_total_bond"].idxmax()]
        self.dataframe.drop(self.removed_dataframe.index, inplace=True)



    def add_validator_count(self):
        """
        This function adds the validator count to the processed_snapshot.
        :param processed_snapshot:
        :return:
        """

        # this all goes into preprocessing, just for now here

        value_counts = self.dataframe.loc[:, 'validator'].value_counts()
        for validator in value_counts.index:
            self.dataframe.loc[self.dataframe['validator'] == validator, 'validator_frequency_current_era'] = value_counts[validator]


    def add_overall_proportional_bond(self):
        """
        This function introduces a new column "overall_total_bond" which is the sum of the proportional bonds of all the
        validators in the current era.
        :param df:
        :return:
        """
        self.dataframe["overall_proportional_bond"] = self.dataframe.groupby("validator")["proportional_bond"].transform("sum")

    def add_overall_total_bond(self):
        """
        This function introduces a new column "overall_total_bond" which is the sum of the total bonds of all the
        validators in the current era.
        :param df:
        :return:
        """
        self.dataframe["overall_total_bond"] = self.dataframe.groupby("validator")["total_bond"].transform("sum")

    def add_expected_sum_stake(self):
        """
        This function adds a column "expected_sum_stake" which is derived by predicting the total solution stake of a
        validator in the current era.
        :param df:
        :return:
        """
        prediction = pd.read_csv(f"./data/model_2/predictions_{self.era}.csv")
        self.dataframe['expected_sum_stake'] = self.dataframe['validator'].map(prediction.set_index('validator')['prediction'])


    def save_dataframe(self):
        """
        This function saves the dataframe to a csv file.
        :param df:
        :return:
        """
        self.dataframe.to_csv(f"{self.output_path}/processed_data_grouped_{self.era}.csv", index=False)
        #self.removed_dataframe.to_csv(f"{self.output_path}/removed_data_{self.era}.csv", index=False)


    def update_solution_bond(self):
        """
        This function updates the solution bond to the distribution calculated in the linear programming approach
        :return:
        """
        calculated_solution = pd.read_csv(f"./data/model_2/{self.era}_max_min_stake.csv")
        calculated_solution.drop("Unnamed: 0", axis=1, inplace=True)
        voter_preferences = pd.read_csv(f"./data/model_2/{self.era}_voter_preferences.csv")
        voter_preferences = voter_preferences.set_index(voter_preferences.columns[0])
        voter_preferences = voter_preferences.stack().reset_index()
        voter_preferences.rename(
            columns={
                "Unnamed: 0": "nominator",
                "level_1": "validator",
                0: "proportional_bond",
            },
            inplace=True,
        )
        calculated_solution = calculated_solution.stack().reset_index()
        calculated_solution.rename(columns={0: "solution_bond"}, inplace=True)
        total_dataframe = pd.concat([voter_preferences, calculated_solution], axis=1)
        total_dataframe = total_dataframe.drop(columns=["level_0", "level_1"])
        total_dataframe = total_dataframe[
            total_dataframe["proportional_bond"] >= 1
        ]
        self.dataframe['solution_bond'] = self.dataframe['validator'].map(total_dataframe.set_index(['nominator', 'validator'])['solution_bond'])

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.process_data()
    preprocessor.save_data()
