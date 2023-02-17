import pandas as pd
import json


class Preprocessor:
    def __init__(self):
        self.voters_dataframes = None
        self.targets_dataframes = None
        self.full_targets_dataframes = None

    def load_snapshot_data(self, snapshot):
        self.voters_dataframes = []
        self.targets_dataframes = []
        for index, file in enumerate(snapshot):
            with open(file) as json_file:
                data = json.load(json_file)
                voters = pd.DataFrame(data['voters'])
                voters.columns = ['Voter', 'Weight', 'Targets']
                self.voters_dataframes.append(voters)
                targets = pd.DataFrame(data['targets'])
                targets.columns = ['Targets']
                self.targets_dataframes.append(targets)

    def load_solution_data(self):
        self.full_targets_dataframes = []
        for index, file in enumerate(self.solutions):
            with open(file) as json_file:
                data = json.load(json_file)
                self.process_solution_data(data)

    @staticmethod
    def process_snapshot_data(snapshot, era):
        # todo: potentially add commission // era points
        """
        place all in Dataframe with columns (ValidatorAddress, TotalBond, ProportionalBond, *[Nominators], NominatorCount,
                                        ElectedCurrentEra, ElectedCounter, selfStake, avgStakePerNominator,)
        :return: Dataframe
        """
        voters = snapshot['voters']
        targets = snapshot['targets']
        processed_dict = {}
        for row in voters:
            for validator in row[2]:
                try:
                    current_values                  = processed_dict[validator]

                    total_bond                      = current_values[0] + row[1]
                    proportional_bond               = current_values[1] + row[1]/len(row[2])
                    list_of_nominators              = current_values[2]
                    list_of_nominators.add(row[0])
                    nominator_count                 = current_values[3] + 1
                    elected_current_era             = 0  # get with solution
                    elected_previous_era            = 0
                    elected_counter                 = 0  # get with solution
                    self_stake                      = 0  # potentially add later/should not give extra information
                    avg_stake_per_nominator         = proportional_bond / len(list_of_nominators)  # todo: adapt metric.
                    era                             = era
                    processed_dict[validator]       = [total_bond, proportional_bond, list_of_nominators,
                                                       nominator_count, elected_current_era, elected_previous_era,
                                                       elected_counter,self_stake, avg_stake_per_nominator, era]
                except KeyError:
                    total_bond                      = row[1]
                    proportional_bond               = row[1]/len(row[2])
                    list_of_nominators              = set()
                    list_of_nominators.add(row[0])
                    nominator_count                 = 1
                    elected_current_era             = 0
                    elected_previous_era            = 0
                    elected_counter                 = 0
                    self_stake                      = 0
                    avg_stake_per_nominator         = row[1]
                    era                             = era
                    processed_dict[validator]       = [total_bond, proportional_bond, list_of_nominators,
                                                       nominator_count, elected_current_era, elected_previous_era,
                                                       elected_counter, self_stake, avg_stake_per_nominator, era]
        return processed_dict

    def update_values(self):
        pass

    @staticmethod
    def process_solution_data(solution, processed_snapshot_dict, previous_snapshot_dict=None):
        # update processed_dict from snapshot data by checking whether validator got elected and increase counter if so.
        for row in solution:
            for winner in row[1]:
                processed_snapshot_dict[winner[0]][4] = 1
                try:
                    if previous_snapshot_dict is not None:
                        processed_snapshot_dict[winner[0]][6] = previous_snapshot_dict[winner[0]][6] + 1
                        processed_snapshot_dict[winner[0]][5] = previous_snapshot_dict[winner[0]][4]
                except KeyError:
                    continue
        return processed_snapshot_dict

    def save_data(self, file_name='processed_data.parquet'):
        self.data.to_parquet(file_name, index=False)

    @staticmethod
    def return_mapping_from_address_to_index(snapshot):
        validator_mapping_dict = {}
        for index, row in enumerate(snapshot['targets']):
            validator_mapping_dict[row] = str(index)
        nominator_mapping_dict = {}
        fail_counter = 0
        for index, row in enumerate(snapshot['voters']):
            nominator_mapping_dict[row[0]] = str(index)
            validator_indices = []
            for validator_address in snapshot['voters'][index][2]:
                try:
                    validator_indices.append(validator_mapping_dict[validator_address])
                except KeyError:
                    fail_counter += 1
                    # print(f"validator_address: {validator_address} is not in available targets")
        return nominator_mapping_dict, validator_mapping_dict

    @staticmethod
    def map_address_to_index(snapshot): # todo: refactor to work with mappings saved
        validator_mapping_dict = {}
        for index, row in enumerate(snapshot['targets']):
            validator_mapping_dict[row] = str(index)
            snapshot['targets'][index] = str(index)
        nominator_mapping_dict = {}
        fail_counter = 0
        for index, row in enumerate(snapshot['voters']):
            nominator_mapping_dict[row[0]] = str(index)
            validator_indices = []
            for validator_address in snapshot['voters'][index][2]:
                try:
                    validator_indices.append(validator_mapping_dict[validator_address])
                except KeyError:
                    fail_counter += 1
                    print(f"validator_address: {validator_address} is not in available targets")
            snapshot['voters'][index] = (str(index),
                                         snapshot['voters'][index][1],
                                         validator_indices)

        return snapshot, nominator_mapping_dict, validator_mapping_dict


if __name__ == "__main__":
    print()
    """
    snapshot_data_files = ['../snapshot_data/590_custom_snapshot.json',
                           '../snapshot_data/591_custom_snapshot.json']
    solution_data_files = ['../storedsolutions/590_0_storedsolution_.json',
                           '../storedsolutions/591_0_storedsolution_.json']
    processor = Preprocessor(snapshots=snapshot_data_files, solutions=solution_data_files)
    processor.process_data()
    processor.save_data()
    """
