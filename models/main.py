import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.adjustment import AdjustmentTool
from src.score import ScoringTool, ScoringUtility
from src.model import Model


def prepare(path=None, target_column="solution_bond", features=None):
    """
    prepare data for training
    :return: model
    """
    # read training data
    dataframe = pd.read_csv(
        path
    )
    model = Model(dataframe, target_column, features)

    return model


def adjust(predicted_dataframe):
    """
    adjust predictions to 100%
    :return: adjusted dataframe
    """
    adj = AdjustmentTool()
    return adj.proportional_split_strategy(predicted_dataframe)


def score(adjusted_predicted_dataframe):
    """
    score the adjusted predictions
    :return: score
    """
    scorer = ScoringTool()
    predicted_dataframe = scorer.dataframe_groupby_predictions_by_validators(
        adjusted_predicted_dataframe
    )
    return scorer.score_solution(predicted_dataframe["prediction"])


def plot_comparison(score_of_prediction, score_of_calculated):
    """
    plot the comparison of the scores
    :param score_of_prediction:
    :param score_of_calculated:
    """
    title_list = ["max(minStake)", "max(sumStake)", "min(varStake)"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    for i in range(3):
        ax = axes[i]
        ax.bar(
            ["Predicted Solution", "Calculated Solution"],
            [score_of_prediction[i], score_of_calculated[i]],
        )
        ax.set_title(title_list[i])

    plt.show()


def compare(score_of_prediction, era, path):
    """
    compare the score of the prediction to the score of the stored solution
    :param score_of_prediction:
    :param era:
    """
    # todo: should pull the stored solution via storage query
    comparer = ScoringUtility()

    filename = (
        path + str(era) + "_stored_solution.json"
    )
    try:
        with open(
            filename,
            "r",
        ) as jsonfile:
            score_of_stored = json.load(jsonfile)["raw_solution"]["score"]
    except FileNotFoundError:
        print("No stored solution found")
        score_of_stored = [0, 0, 0]
    return (
        comparer.is_score1_better_than_score2(
            score_of_prediction, score_of_stored
        ),
        score_of_prediction,
        score_of_stored,
    )


def main(args):
    """
    prepare data, train model, predict, adjust, score
    :param era: era to test on
    :return: comparison of scores
    """
    if args.train is None:
        print("Model is not trained")
        # todo: THIS IS INCOMPLETE // must provide X_test via json
        model = Model.load_trained_model(args.model)
    else:
        model = prepare(args.train, args.target, args.features)
        model.model_selection(args.model)
        model.split_data(args.era)
        model.scale_data()
        model.model.fit(model.X_train, model.y_train)
        print(f"model {args.model} trained")

    if args.save:
        model.save_trained_model()

    predicted_dataframe = pd.concat([model.X[model.X["era"] == args.era], model.y[model.X["era"] == args.era]], axis=1)
    predicted_dataframe["prediction"] = model.model.predict(model.X_test)
    predicted_dataframe["prediction"] = predicted_dataframe[
        "prediction"
    ].astype(int)
    print("predictions made")

    # adjust predictions to 100%
    adjusted_predicted_dataframe = adjust(predicted_dataframe)
    print("predictions adjusted")

    # score predictions
    score_of_prediction = score(adjusted_predicted_dataframe)

    result, score_of_prediction, score_of_calculated = compare(
        score_of_prediction, args.era, args.compare
    )

    if args.plot:
        plot_comparison(score_of_prediction, score_of_calculated)

    print(f"result: {result}")
    print(f"score of prediction: {score_of_prediction}")
    print(f"score of stored: {score_of_calculated}")


def setup():
    """
    setup parser and override any default arguments provided in config.json
    :return:
    """
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    parser.add_argument("-e", "--era", type=int)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="linear, gradientboosting, randomforest, ridge, lasso",
    )
    parser.add_argument("-t", "--train", type=str, help="provide path to training data csv")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-x", "--target", type=str, help="target column")
    parser.add_argument("-f", "--features", nargs="+", help="list of features")
    parser.add_argument("-c", "--compare", type=str, help="provide path to stored solution")
    return parser


if __name__ == "__main__":
    parser = setup()
    main(
        parser.parse_args()
    )
