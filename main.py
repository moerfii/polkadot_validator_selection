import argparse
import json
import os
import time

from data_collection.main import (
    calculate_phragmen,
    get_model_1_data,
    process_model_1_data,
    process_model_2_data,
    process_model_3_data,
)
from models.main import (
    predict_model_1,
    predict_model_2,
    predict_model_3_Xtest,
)


def subscribe():
    """
    This function subscribes to the storage query "snapshot" and saves the data in the data folder
    :return:
    """
    pass



def predict_Xtest(args):
    """
    This function trains the model given "the perfect" active set, predefined by the sequential phragmen algorithm.
    :param args:
    :return:
    """

    print(f"Predicting era: {args.era}")
    predict_model_3_Xtest(args)


def submit():
    """
    This function submits the solution to the blockchain
    :return:
    """
    pass


def check_data(args):
    """
    This function checks whether the snapshots required for preprocessing are available
    :param args:
    :return:
    """
    range_required = range(args.era - 6, args.era + 1)  # set_era_range(test_era)
    list_of_eras = []
    list_of_eras.extend(range_required)
    snapshots_available = os.listdir("data_collection/data/snapshot_data/")
    snapshots_available = [int(x.split("_")[0]) for x in snapshots_available]
    snapshots_available.sort()
    eras_to_acquire = []
    for era in list_of_eras:
        if era not in snapshots_available:
            eras_to_acquire.append(era)
    return eras_to_acquire


def check_phragmen(args):
    """
    This function checks whether the phragmen solution is available
    :param args:
    :return:
    """
    eras_to_acquire = []
    range_required = range(args.era - 9, args.era + 1)  # set_era_range(test_era)
    list_of_eras = []
    list_of_eras.extend(range_required)
    snapshots_available = os.listdir("data_collection/data/calculated_solutions_data/")
    snapshots_available = [int(x.split("_")[0]) for x in snapshots_available]
    snapshots_available.sort()
    for era in list_of_eras:
        if era not in snapshots_available:
            eras_to_acquire.append(era)
    return eras_to_acquire


def prepare(args):
    # check if data is available
    eras_to_acquire = check_data(args)
    if not len(eras_to_acquire):
        print(f"Data required for test era: {args.era} is available")
    else:
        # if not, gather data
        for era in eras_to_acquire:
            print(
                f"Data for era: {era} is not available, gathering data. Ensure that the snapshot is available"
            )
            get_model_1_data(era)
            print(f"Data for era: {era} gathered")

    phragmens_available = check_phragmen(args)
    if not len(phragmens_available):
        print(f"Phragmen solution required for test era: {args.era} is available")
    else:
        # if not, gather data
        for era in phragmens_available:
            print(f"Phragmen solution for era: {era} is not available, calculating")
            start = time.time()
            calculate_phragmen(era)
            print(f"Phragmen solution for era: {era} gathered")
            end = time.time()
            print(f"Time taken: {end-start}")

    # when data available, preprocess model 1
    process_model_1_data(args)
    print("Model 1 preprocessing complete")
    # predict model 1 (probability if selected)
    predict_model_1(args)
    print("Model 1 prediction complete")
    # preprocess model 2
    process_model_2_data(args)
    print("Model 2 preprocessing complete")
    # predict model 2 (global distribution of stake)
    predict_model_2(args)
    print("Model 2 prediction complete")
    # preprocess model 3
    process_model_3_data(args)
    print("Model 3 preprocessing complete")
    # done


def main(args):
    eras = range(args.era - 30, args.era + 1)
    for era in eras:
        args.era = era
        #prepare(args)
        predict_Xtest(args)


def setup():

    config = "config.json"
    with open(config, "r") as jsonfile:
        config = json.load(jsonfile)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    parser.add_argument("-e", "--era", help="era to test on")
    parser.add_argument("-m1", "--model_1", help="select model 1")
    parser.add_argument("-m1p", "--model_1_path", help="path to data model 1")
    parser.add_argument(
        "-f1", "--features_1", nargs="+", help="list of features for model 1"
    )
    parser.add_argument("-t1", "--target_1", help="target column for model 1")
    return parser


if __name__ == "__main__":
    parser = setup()
    main(parser.parse_args())
