import json
import sys

import pandas as pd


def set_era_range(test_era):
    test_era = int(test_era)
    range_required = range(test_era - 365, test_era + 1)
    list_of_eras = []
    list_of_eras.extend(range_required)
    return list_of_eras


def read_json(path_to_json):
    with open(path_to_json, "r") as jsonfile:
        return json.load(jsonfile), jsonfile


def save_json(path, object):
    with open(path, "w", encoding="utf-8") as jsonfile:
        jsonfile.write(json.dumps(object, ensure_ascii=False, indent=4))
        jsonfile.close()


def read_parquet(path_to_parquet):
    return pd.read_parquet(path_to_parquet)


def get_era(self, block_number):
    self.snapshot_instance.set_block_number(block_number)
    return self.snapshot_instance.get_era()


@staticmethod
def save_file(dataframe):
    dataframe.to_parquet("../block_numbers/new_block_numbers_dataframe.parquet")


def progress_of_loop(counter, total, name):
    counter += 1
    progress = int((counter / len(total)) * 100)
    sys.stdout.write(f"Progress of {name}: {progress}%   \r")
    sys.stdout.flush()
    return counter


def partition_into_batches(array, maxbatchsize):
    return [
        array[i * maxbatchsize : (i + 1) * maxbatchsize]
        for i in range((len(array) + maxbatchsize - 1) // maxbatchsize)
    ]


if __name__ == "__main__":
    # snapshot = StakingSnapshot()
    print()
