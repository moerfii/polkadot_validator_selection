import json
import sys

import pandas as pd


def read_json(path_to_json):
    with open(path_to_json, "r") as jsonfile:
        return json.load(jsonfile), jsonfile


def save_json(path, object):
    with open(path, "w", encoding="utf-8") as jsonfile:
        jsonfile.write(json.dumps(object, ensure_ascii=False, indent=4))
        jsonfile.close()


def read_parquet(path_to_parquet):
    return pd.read_parquet(path_to_parquet)


def group_data(path_to_signedphase_json, path_to_solutionstored_json):
    signedphase_block_numbers = sorted(read_json(path_to_signedphase_json))
    solutionstored_block_numbers = sorted(
        read_json(path_to_solutionstored_json)
    )
    signedphase_era_dict = {}
    solution_era_dict = {}
    for signedphase_block in signedphase_block_numbers:
        print(signedphase_block)
        snapshot.set_block_number(signedphase_block)
        signedphase_era_dict[
            str(snapshot.get_era(signedphase_block)["index"])
        ] = signedphase_block
    for solution_block in solutionstored_block_numbers:
        print(solution_block)
        snapshot.set_block_number(solution_block)
        solution_era_dict[
            str(snapshot.get_era(solution_block)["index"])
        ] = solution_block

    signedphase_eras = set(signedphase_era_dict.keys())
    solution_eras = set(solution_era_dict.keys())
    common_eras = signedphase_eras.intersection(solution_eras)
    full_data_list = []
    for era in common_eras:
        full_data_list.append(
            {
                "Era": era,
                "SignedPhaseBlock": signedphase_era_dict[era],
                "SolutionStoredBlock": solution_era_dict[era],
            }
        )
    return pd.DataFrame(full_data_list)


def get_era(self, block_number):
    self.snapshot_instance.set_block_number(block_number)
    return self.snapshot_instance.get_era()


@staticmethod
def save_file(dataframe):
    dataframe.to_parquet(
        "../block_numbers/new_block_numbers_dataframe.parquet"
    )


def progress_of_loop(counter, total, name):
    counter += 1
    progress = int((counter / len(total)) * 100)
    sys.stdout.write(f"Progress of {name}: {progress}%   \r")
    sys.stdout.flush()
    return counter


if __name__ == "__main__":
    #snapshot = StakingSnapshot()
    df = group_data(
        "../block_numbers/signedphase_blocknumbers.json",
        "../block_numbers/solutionstored_blocknumbers.json",
    )
    save_file(df)
    print()
