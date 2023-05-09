import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from src.adjustment import AdjustmentTool
from src.score import ScoringTool, ScoringUtility
from src.model import Model

from sklearn.compose import make_column_transformer
import pickle
from sklearn.model_selection import GroupShuffleSplit



def split_data(dataframe, test_era=None):
    features = [
        "proportional_bond",
        "total_bond",
        "prev_min_stake",
        "prev_sum_stake",
        "prev_variance_stake",
        "nominator",
        "validator",
        "era",
    ]
    column_transformer = make_column_transformer(
        (
            StandardScaler(),
            [
                "proportional_bond",
                "total_bond",
                "prev_min_stake",
                "prev_sum_stake",
                "prev_variance_stake",
            ],
        ),
        remainder="passthrough",
    )
    if test_era is None:
        test_era = dataframe["era"].iloc[-1]
    # potentially add total proportional bond i.e grouping by validator
    x = dataframe.loc[:, features]

    y = dataframe.loc[:, ["solution_bond", "era"]]
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train = x.loc[dataframe["era"] != test_era]
    column_transformer.fit(X_train)
    X_train = pd.DataFrame(column_transformer.transform(X_train))

    X_test = x.loc[dataframe["era"] == test_era]
    total_bond = X_test["total_bond"]
    X_test = pd.DataFrame(column_transformer.transform(X_test))
    y_train = y.loc[dataframe["era"] != test_era].drop(["era"], axis=1)
    y_test = y.loc[dataframe["era"] == test_era].drop(["era"], axis=1)

    X_test.columns = features
    X_train.columns = features
    X_train.drop(["era"], axis=1, inplace=True)
    X_test.drop(["era"], axis=1, inplace=True)

    return X_train, X_test, y_train, y_test, total_bond


def predict(model, X_test):
    return model.predict(X_test)


def train(model, X_train, y_train):
    model.fit(X_train, y_train.values.ravel())


def save_trained_model(model):
    """
    save trained model
    :return:
    """

    filename = f"../models/trained_models/{type(model).__name__}.sav"
    pickle.dump(model, open(filename, "wb"))


def load_trained_model(model_name):
    """
    load trained model
    :return:
    """
    filename = f"../models/trained_models/{model_name}.sav"
    model = pickle.load(open(filename, "rb"))
    return model


def model_selection(model="randomforest"):
    """
    select model via cli
    :return: model
    """
    if model == "linear":
        return LinearRegression()
    elif model == "gradientboosting":
        return GradientBoostingRegressor()
    elif model == "randomforest":
        return RandomForestRegressor()
    elif model == "ridge":
        return linear_model.Ridge(alpha=0.5)
    elif model == "lasso":
        return linear_model.Lasso(alpha=0.1)


def prepare(test_era=None, path=None, target_column="solution_bond", features=None):
    """
    prepare data for training

    :return:
    """
    # read training data
    dataframe = pd.read_csv(
        path
    )
    model = Model(dataframe, target_column, features)

    return model



    """ old code    
    # split data into train and test (provide era to test on) # todo: in the future the test era should be the last era
    X_train, X_test, y_train, y_test, total_bond = split_data(
        dataframe, test_era=test_era
    )

    # prepare predicted dataframe
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    predicted_dataframe = pd.concat([X_test, y_test], axis=1)
    predicted_dataframe["total_bond"] = total_bond.reset_index(drop=True)
    drop_columns = ["nominator", "validator"]
    X_train.drop(drop_columns, axis=1, inplace=True)
    X_test.drop(drop_columns, axis=1, inplace=True)
    return X_train, X_test, y_train, y_test, predicted_dataframe
    """

def adjust(predicted_dataframe):
    """
    adjust predictions to 100%
    :return:
    """
    adj = AdjustmentTool()
    return adj.even_split_strategy(predicted_dataframe)


def score(adjusted_predicted_dataframe):
    scorer = ScoringTool()
    predicted_dataframe = scorer.dataframe_groupby_predictions_by_validators(
        adjusted_predicted_dataframe
    )
    return scorer.score_solution(predicted_dataframe["prediction"])


def plot_comparison(score_of_prediction, score_of_calculated):
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


def compare(score_of_prediction, era):
    # todo: should pull the stored solution via storage query
    # todo: should calculate the score of the stored solution
    comparer = ScoringUtility()
    filename = (
        f"../data_collection/data/calculated_solutions_data/{era}_winners.json"
    )
    with open(
        filename,
        "r",
    ) as jsonfile:
        stored_solution = json.load(jsonfile)
    score_of_calculated = comparer.calculate_score(stored_solution)
    return (
        comparer.is_score1_better_than_score2(
            score_of_prediction, score_of_calculated
        ),
        score_of_prediction,
        score_of_calculated,
    )


def main(args):
    """
    prepare data, train model, predict, adjust, score
    :param era: era to test on
    :return:
    """

    if args.train is None:
        print("Model is not trained")
        # todo: THIS IS INCOMPLETE
        model = load_trained_model(args.model)

        # X_test = pd.read_csv(
        #    f"../data_collection/data/model_2/{era}.csv"
        # )  # todo: adapt to last era/
        # should probably not be csv
    else:
        model = prepare(args.era, args.train, args.target, args.features)
        model.model_selection(args.model)
        model.cross_validate()
        print("model trained")

        """
        ########3 old
        # prepare data
        X_train, X_test, y_train, y_test, predicted_dataframe = prepare(
            args.era, args.train, args.target
        )

        print("data prepared")

        # select model
        model = model_selection(model)

        # train model
        train(model, X_train, y_train)
        print("model trained")
        """
    if args.save:
        save_trained_model(model)
    # predict & append to dataframe
    predicted_dataframe["prediction"] = predict(model, X_test)

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
        score_of_prediction, args.era
    )

    if args.plot:
        plot_comparison(score_of_prediction, score_of_calculated)

    return compare(score_of_prediction, args.era)


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
    return parser


if __name__ == "__main__":
    parser = setup()
    result, score_of_prediction, score_of_calculated = main(
        parser.parse_args()
    )
    print(result)
    print(score_of_prediction)
    print(score_of_calculated)
