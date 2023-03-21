import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from src.adjustment import AdjustmentTool
from data_collection.src.score_data import ScoringUtility
import json
from src.score import ScoringTool


def prepare_data(dataframe, test_era):
    features = [
        "nominator",
        "validator",
        "proportional_bond",
        "total_bond",
        "number_of_validators",
        "total_proportional_bond",
        "era",
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
    model.fit(X_train, y_train)


def model_selection(model):
    """
    select model via cli
    :return: model
    """
    if model == "linear":
        return LinearRegression()
    elif model == "logistic":
        return LogisticRegression()
    elif model == "randomforest":
        return RandomForestRegression()
    elif model == "decisiontree":
        return DecisionTreeClassifier()
    elif model == "knn":
        return KNeighborsClassifier()
    elif model == "svm":
        return SVC()
    elif model == "naivebayes":
        return GaussianNB()
    elif model == "perceptron":
        return Perceptron()
    elif model == "sgd":
        return SGDClassifier()
    elif model == "linear":
        return LinearSVC()
    elif model == "mlp":
        return MLPClassifier()
    elif model == "ridge":
        return RidgeClassifier()
    elif model == "sgd":
        return SGDClassifier()
    elif model == "passiveaggressive":
        return PassiveAggressiveClassifier()


def prepare(*args):
    """
    prepare data for training
    :return:
    """
    # read training data
    df1 = pd.read_csv("../data_collection/df_bond_distribution_0.csv")
    # df2 = pd.read_csv("../data_collection/df_bond_distribution_1.csv")
    # df3 = pd.read_csv("../data_collection/df_bond_distribution_2.csv")
    # df = pd.concat([df1, df2, df3])
    # df.rename(columns={"Unnamed: 0": "nominator"}, inplace=True)

    # split data into train and test (provide era to test on) # todo: in the future the test era should be the last era
    X_train, X_test, y_train, y_test = prepare_data(df1, test_era=739)

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


def compare(score_of_prediction, era):
    # todo: should pull the calculated solution via storage query
    comparer = ScoringUtility()
    with open(
        "../data_collection/data/calculated_solutions_data/739_winners.json",
        "r",
    ) as jsonfile:
        stored_solution = json.load(jsonfile)
    score_of_calculated = comparer.calculate_score(stored_solution)
    return comparer.is_score1_better_than_score2(
        score_of_prediction, score_of_calculated
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

    # select model
    model = model_selection(model)

    # train model
    train(model, X_train, y_train)

    # predict & append to dataframe
    predicted_dataframe["prediction"] = predict(model, X_test)

    # adjust predictions to 100%
    adjusted_predicted_dataframe = adjust(predicted_dataframe)

    # score predictions
    score_of_prediction = score(adjusted_predicted_dataframe)
    return compare(score_of_prediction, era)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--era", type=int, default=739)
    main(parser.parse_args())
