import argparse
import json
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from .src.adjustment import AdjustmentTool
from .src.score import ScoringTool, ScoringUtility
from .src.model import Model
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


def predict_model_1(args):
    """
    predict model 1: This model predicts the probability of being selected for the next active set.
    :return: predicted dataframe
    """

    eras = range(args.era - 6, args.era + 1)
    for era in eras:
        model = prepare(
            path=args.model_1_path,
            target_column=args.target_1,
            features=args.features_1,
            era=era,
        )
        model.model_selection(args.model_1)
        model.split_data(era)
        model.scale_data()
        if args.model_1 == "lgbm_classifier":
            model.model.fit(model.X_train, model.y_train, eval_set=[(model.X_test, model.y_test)],
                            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                            )
        else:
            model.model.fit(model.X_train, model.y_train)

        print(f"Model 1 {era} score: {model.model.score(model.X_test, model.y_test)}")
        predictions = model.model.predict_proba(model.X_test)
        predicted_dataframe = pd.concat(
            [
                model.dataframe[model.X["era"] == era],
                model.y[model.X["era"] == era],
            ],
            axis=1,
        )
        predicted_dataframe["prediction"] = predictions[:, 1]
        predicted_dataframe.to_csv(
            args.intermediate_results_path + f"{era}_model_1_predictions.csv",
            index=False,
        )


def predict_model_2(args):
    """
    predict model 2: This model predicts the global distribution of stake.
    :param args:
    :return:
    """

    eras = range(args.era - 3, args.era + 1)
    for era in eras:
        model = prepare(
            path=args.model_2_path + "_grouped",
            target_column=args.target_2,
            features=args.features_2,
            era=era,
        )
        model.split_data(era)
        if args.model_2_load is not None:
            model.model = pickle.load(open(args.model_2_load, "rb"))
            model.column_transformer = pickle.load(open(args.scaler_2_load, "rb"))
            model.scale_data()

        else:
            model.model_selection(args.model_2)
            model.scale_data()
            if args.model_2 == "lgbm_model_2":
                model.model.fit(model.X_train, model.y_train, eval_set=[(model.X_test, model.y_test)],
                                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                                )
            else:
                model.model.fit(model.X_train, model.y_train)
        # print(f"Model 2 Xtest score: {model.model.score(X_test, model.y_test)}")
        print(f"Model 2 {era} score: {model.model.score(model.X_test, model.y_test)}")

        predictions = model.model.predict(model.X_test)

        predicted_dataframe = pd.concat(
            [model.X[model.X["era"] == era], model.y[model.X["era"] == era]],
            axis=1,
        )  # model.dataframe?

        predicted_dataframe["prediction"] = predictions

        predicted_dataframe.to_csv(
            args.intermediate_results_path + f"{era}_model_2_predictions.csv",
            index=False,
        )

    model = prepare(
        path=args.model_2_path + "_grouped",
        target_column=args.target_2,
        features=args.features_2,
        era=args.era,
    )
    model.model_selection(args.model_2)
    model.split_data(args.era)
    model.scale_data()
    model.model.fit(model.X_train, model.y_train)
    X = pd.read_csv(
        f"data_collection/data/processed_data/model_2_data_grouped_Xtest_{args.era}.csv"
    )
    # drop non numeric_columns
    Xtest = X.loc[:, args.features_2]
    drop_columns = Xtest.select_dtypes(include=["object"]).columns
    Xtest = Xtest.drop(drop_columns, axis=1)
    Xtest = Xtest.drop("era", axis=1)
    X_test = model.column_transformer.transform(Xtest)

    predictions_X_test = model.model.predict(X_test)
    X["prediction"] = predictions_X_test
    # sort values by predictions and keep top 297 rows
    X.to_csv(
        args.intermediate_results_path + f"{args.era}_model_2_X_test_predictions.csv",
        index=False,
    )


def predict_model_3_Xtest(args):
    """
    predict model 3: This model predicts the stake of each validator. However the active set was predicted by model 1.
    :param args:
    :param era:
    :return:
    """
    model = prepare(
        path=args.model_3_path,
        target_column=args.target_3,
        features=args.features_3,
        era=args.era,
    )
    model.model_selection(args.model_3)

    #model.divide_target_by_total_bond()
    model.model_selection(args.model_3)
    #model.feature_selection()
    model.split_data(args.era)
    model.scale_data()
    if args.model_3 == "lgbm_model_3":
        model.model.fit(model.X_train, model.y_train, eval_set=[(model.X_test, model.y_test)],
                        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                        )
    else:
        model.model.fit(model.X_train, model.y_train)

    X = pd.read_csv(
        f"data_collection/data/processed_data/model_3_data_Xtest_{args.era}.csv"
    )
    # drop non numeric_columns
    Xtest = X.loc[:, args.features_3]
    drop_columns = Xtest.select_dtypes(include=["object"]).columns
    X_test = Xtest.drop(drop_columns, axis=1)
    X_test = X_test.drop("era", axis=1)
    X_test = model.column_transformer.transform(X_test)

    predicted_dataframe = Xtest
    predicted_dataframe["prediction"] = model.model.predict(X_test)

    predicted_dataframe["prediction"] = predicted_dataframe["prediction"].astype(int)
    print("predictions made")

    adjusted_predicted_dataframe = adjust(predicted_dataframe, args)
    print("predictions adjusted")

    # score predictions
    score_of_prediction = score(adjusted_predicted_dataframe)

    result, score_of_prediction, score_of_calculated = compare_with_phragmen(
        score_of_prediction, args.era, args.compare
    )

    log_score(
        score_of_prediction=score_of_prediction,
        score_of_calculated=score_of_calculated,
        era=args.era,
        model=args.model_3,
        result=result,
    )

    if args.plot:
        plot_comparison(score_of_prediction, score_of_calculated)

    print(f"result: {result}")
    print(f"score of prediction: {score_of_prediction}")
    print(f"score of stored:     {score_of_calculated}")

    adjusted_predicted_dataframe.to_csv(
        args.intermediate_results_path + f"{args.era}_model_3_predictions_sequentialadj_final.csv",
        index=False,
    )


def prepare(path=None, target_column="solution_bond", features=None, era=None):
    """
    prepare data for training
    :return: model
    """

    eras = range(era - 3, era + 1)
    dataframes = []
    for era in eras:
        tmp_dataframe = pd.read_csv(path + f"_{era}.csv")
        dataframes.append(tmp_dataframe)
    dataframe = pd.concat(dataframes, ignore_index=True)
    model = Model(dataframe, target_column, features)
    return model


def adjust(predicted_dataframe, args):
    """
    adjust predictions to 100%
    :return: adjusted dataframe
    """
    adj = AdjustmentTool()
    return adj.proportional_split_strategy(predicted_dataframe, args)


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
        ax.set_yscale("log")
    fig.tight_layout()

    plt.show()


def compare_with_phragmen(score_of_prediction, era, path):
    """
    compare the score of the prediction to the score of the calculated solution
    :param score_of_prediction:
    :param era:
    """
    comparer = ScoringUtility()

    filename = path + str(era) + "_winners.json"

    try:
        with open(
            filename,
            "r",
        ) as jsonfile:
            score_of_calculated = json.load(jsonfile)
            score_of_calculated = list(comparer.calculate_score(score_of_calculated))
    except FileNotFoundError:
        print("No calculated solution found, attempt to pull from storage")
        score_of_calculated = [0, 0, 0]

    return (
        comparer.is_score1_better_than_score2(score_of_prediction, score_of_calculated),
        score_of_prediction,
        score_of_calculated,
    )


def compare(score_of_prediction, era, path):
    """
    compare the score of the prediction to the score of the stored solution
    :param score_of_prediction:
    :param era:
    """
    # todo: should pull the stored solution via storage query
    comparer = ScoringUtility()

    filename = path + str(era) + "_stored_solution.json"
    try:
        with open(
            filename,
            "r",
        ) as jsonfile:
            score_of_stored = json.load(jsonfile)["raw_solution"]["score"]
            if isinstance(score_of_stored, dict):
                score_of_stored = list(score_of_stored.values())
    except FileNotFoundError:
        print("No stored solution found, attempt to pull from storage")
        score_of_stored = [0, 0, 0]

    return (
        comparer.is_score1_better_than_score2(score_of_prediction, score_of_stored),
        score_of_prediction,
        score_of_stored,
    )


def log_score(
    score_of_prediction,
    score_of_calculated,
    era,
    model,
    normalized_error=None,
    score_model=None,
    result=None,
):
    """
    log the score of the prediction
    :param score_of_prediction:
    :param era:
    """
    log = {
        "era": era,
        "model": model,
        "score_prediction": score_of_prediction.tolist(),
        "score_stored": score_of_calculated,
        "normalized_error": normalized_error,
        "score_model": score_model,
        "result": result,
    }
    # write to new file
    with open(f"./models/results/{model}_{era}_log_sequentialadj_final.json", "w") as jsonfile:
        json.dump(log, jsonfile, indent=4)


def main(args):
    """
    prepare data, train model, predict, adjust, score
    :param era: era to test on
    :return: comparison of scores
    """
    print(f"era: {args.era}")
    if args.train is None:
        print("Model is not trained")
        # todo: THIS IS INCOMPLETE // must provide X_test via json
        model = Model.load_trained_model(args.model)
    else:
        model = prepare(args.train, args.target, args.features)
        model.divide_target_by_total_bond()
        model.model_selection(args.model)
        model.split_data(args.era)
        model.scale_data()
        # model.feature_selection()
        model.model.fit(model.X_train, model.y_train)


    if args.save:
        model.save_trained_model()

    predicted_dataframe = pd.concat(
        [
            model.X[model.X["era"] == args.era],
            model.y[model.X["era"] == args.era],
        ],
        axis=1,
    )
    predicted_dataframe["prediction"] = model.model.predict(model.X_test)
    error = mean_squared_error(
        model.model.predict(model.X_test), model.y_test, squared=False
    )
    normalized_error = error / (
        predicted_dataframe.loc[:, "prediction"].max()
        - predicted_dataframe.loc[:, "prediction"].min()
    )
    print(f"normalized error: {normalized_error}")
    predicted_dataframe["prediction"] = model.multiply_predictions_by_total_bond(
        predicted_dataframe
    )
    predicted_dataframe["prediction"] = predicted_dataframe["prediction"].astype(int)
    print("predictions made")

    # adjust predictions to 100%
    adjusted_predicted_dataframe = adjust(predicted_dataframe)
    print("predictions adjusted")

    # score predictions
    score_of_prediction = score(adjusted_predicted_dataframe)

    log_score(score_of_prediction, args.era, args.model, normalized_error)

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
    parser.add_argument(
        "-t", "--train", type=str, help="provide path to training data csv"
    )
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-x", "--target", type=str, help="target column")
    parser.add_argument("-f", "--features", nargs="+", help="list of features")
    parser.add_argument(
        "-c", "--compare", type=str, help="provide path to stored solution"
    )
    return parser


if __name__ == "__main__":
    parser = setup()
    main(parser.parse_args())
