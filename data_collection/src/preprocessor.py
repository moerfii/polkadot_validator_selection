import pandas as pd
import json


class Preprocessor:
    def __init__(self, snapshots, solutions):
        self.snapshots = snapshots
        self.solutions = solutions
        self.voters_dataframes = None
        self.targets_dataframes = None
        self.full_targets_dataframes = None

        self.load_snapshot_data()
        self.load_solution_data()

    def load_snapshot_data(self):
        self.voters_dataframes = []
        self.targets_dataframes = []
        for index, file in enumerate(self.snapshots):
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


    def process_snapshot_data(self):
        # count how many (weighted) votes validator got?
        pass

    def process_solution_data(self, data):
        # create column of address(index) with 1' and 0's depending if they got elected into active set.
        pass

    def save_data(self, file_name='processed_data.parquet'):
        self.data.to_parquet(file_name, index=False)


if __name__ == "__main__":
    snapshot_data_files = ['../snapshot_data/590_custom_snapshot.json',
                           '../snapshot_data/591_custom_snapshot.json']
    solution_data_files = ['../storedsolutions/590_0_storedsolution_.json',
                           '../storedsolutions/591_0_storedsolution_.json']
    processor = Preprocessor(snapshots=snapshot_data_files, solutions=solution_data_files)
    processor.process_data()
    processor.save_data()
