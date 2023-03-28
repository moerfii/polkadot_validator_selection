import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model

from src.adjustment import AdjustmentTool
from src.score import ScoringTool, ScoringUtility


def prepare_data(dataframe, test_era):
    features = [
        "nominator",
        "validator",
        "proportional_bond",
        "total_bond",
        "number_of_validators",
        "total_proportional_bond",
        "era",
        "prev_min_stake",
        "prev_sum_stake",
        "prev_variance_stake",
    ]  # potentially add total proportional bond i.e grouping by validator
    x = dataframe.loc[:, features]
    # x_scaled = StandardScaler().fit_transform(x)
    y = dataframe.loc[:, ["solution_bond", "era"]]
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train = x.loc[dataframe["era"] != test_era].drop("era", axis=1)
    X_test = x.loc[dataframe["era"] == test_era].drop("era", axis=1)
    y_train = y.loc[dataframe["era"] != test_era].drop("era", axis=1)
    y_test = y.loc[dataframe["era"] == test_era].drop("era", axis=1)
    return X_train, X_test, y_train, y_test


def predict(model, X_test):
    return model.predict(X_test)


def train(model, X_train, y_train):
    model.fit(X_train, y_train.values.ravel())


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



def prepare(*args):
    """
    prepare data for training
    :return:
    """
    # read training data
    df1 = pd.read_csv(
        "../data_collection/data/model_2/df_bond_distribution_0.csv"
    )
    df2 = pd.read_csv(
        "../data_collection/data/model_2/df_bond_distribution_1.csv"
    )
    # df3 = pd.read_csv("../data_collection/df_bond_distribution_2.csv")
    df = pd.concat([df1, df2])
    # df.rename(columns={"Unnamed: 0": "nominator"}, inplace=True)

    # split data into train and test (provide era to test on) # todo: in the future the test era should be the last era
    X_train, X_test, y_train, y_test = prepare_data(df, test_era=739)

    # prepare predicted dataframe
    predicted_dataframe = pd.concat([X_test, y_test], axis=1)
    drop_columns = ["nominator", "validator"]
    X_train.drop(drop_columns, axis=1, inplace=True)
    X_test.drop(drop_columns, axis=1, inplace=True)
    return X_train, X_test, y_train, y_test, predicted_dataframe


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
    # todo: should pull the calculated solution via storage query
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
    if args.era is None:
        era = 739
    else:
        era = args.era
    if args.model is None:
        model = "linear"
    else:
        model = args.model

    # prepare data
    X_train, X_test, y_train, y_test, predicted_dataframe = prepare(era)
    print("data prepared")

    # select model
    model = model_selection(model)

    # train model
    train(model, X_train, y_train)
    print("model trained")

    # predict & append to dataframe
    predicted_dataframe["prediction"] = predict(model, X_test)
    print("predictions made")

    # adjust predictions to 100%
    adjusted_predicted_dataframe = adjust(predicted_dataframe)
    print("predictions adjusted")

    # score predictions
    score_of_prediction = score(adjusted_predicted_dataframe)

    result, score_of_prediction, score_of_calculated = compare(
        score_of_prediction, era
    )

    if args.plot:
        plot_comparison(score_of_prediction, score_of_calculated)

    return compare(score_of_prediction, era)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--era", type=int, default=739)
    parser.add_argument("-m", "--model", type=str, default="ridge", help="linear, gradientboosting, randomforest, ridge, lasso")
    parser.add_argument("-p", "--plot", action="store_true")
    result, score_of_prediction, score_of_calculated = main(
        parser.parse_args()
    )
    print(result)
    print(score_of_prediction)
    print(score_of_calculated)
