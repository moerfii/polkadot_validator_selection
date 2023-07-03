import argparse
import pickle

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
import pandas as pd
from src.score import ScoringTool
from src.adjustment import AdjustmentTool
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn.svm import SVC


class Model:
    def __init__(self, dataframe, target_column, features):
        self.dataframe = dataframe
        self.target_column = target_column
        self.features = features
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.total_bond = None
        self.preprocess_data()

    def objective(self, trial):
        model_type = trial.suggest_categorical(
            "regressor",
            ["xgboost"] #["gradientboosting","lgbm", "xgboost"]
        )
        if model_type == "randomforest":
            self.model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                random_state=42,
            )
        elif model_type == "gradientboosting":
            self.model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )
        elif model_type == "ridge":
            self.model = Ridge(alpha=trial.suggest_float("alpha", 0.01, 1.0))
        elif model_type == "lasso":
            self.model = Lasso(alpha=trial.suggest_float("alpha", 0.01, 1.0))
        elif model_type == "lgbm":
            self.model = LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )
        elif model_type == "xgboost":
            self.model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )

        elif model_type == "logistic":
            self.model = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 1.0),

            )
        elif model_type == "svc":
            self.model = SVC(
                C=trial.suggest_float("C", 0.01, 1.0),

            )
        elif model_type == "randomforest_classifier":
            self.model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                random_state=42,
            )

        elif model_type == "gradientboosting_classifier":
            self.model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )

        elif model_type == "lbgm_classifier":
            self.model = LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )
        elif model_type == "xgboost_classifier":
            self.model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )


    def objective_model_accuracy(self, trial):
        self.objective(trial)
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return cross_val_score(
            self.model,
            self.X_train,
            self.y_train,
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
            scoring="neg_root_mean_squared_error",
        ).mean()

    def objective_model_classification_time_series(self, trial):
        self.objective(trial)
        return self.cross_validate_time_series_classification()

    def objective_model_accuracy_time_series(self, trial):
        self.objective(trial)
        return self.cross_validate_time_series_accuracy()

    def objective_score_boosting(self, trial):
        self.objective(trial)
        return self.cross_validate_time_series_score()

    @staticmethod
    def adjust(predicted_dataframe):
        """
        adjust predictions to 100%
        :param predicted_dataframe:
        :return:
        """
        adjustment_tool = AdjustmentTool()
        adjusted_predicted_dataframe = (
            adjustment_tool.proportional_split_strategy(predicted_dataframe)
        )
        return adjusted_predicted_dataframe

    @staticmethod
    def score(adjusted_predicted_dataframe):
        scorer = ScoringTool()
        predicted_dataframe = (
            scorer.dataframe_groupby_predictions_by_validators(
                adjusted_predicted_dataframe
            )
        )
        return scorer.score_solution(predicted_dataframe["prediction"])

    def feature_selection(self):
        """
        feature selection
        :return:
        """
        selector = SelectKBest(f_regression, k=5)
        selector.fit(self.X_train, self.y_train)
        self.X_train = selector.transform(self.X_train)
        self.X_test = selector.transform(self.X_test)

    def divide_target_by_total_bond(self):
        """
        divide target by total bond
        :return:
        """
        self.total_bond = self.dataframe["total_bond"]
        self.y = self.y / self.total_bond

    def multiply_predictions_by_total_bond(self, predictions):
        """
        multiply predictions by total bond
        :return:
        """
        predictions = predictions["prediction"]
        predictions = predictions * self.total_bond
        return predictions

    def preprocess_data(self):
        """
        preprocess data for training
        :return: X_train, X_test, y_train, y_test
        """
        self.X = self.dataframe.loc[:, self.features]
        self.y = self.dataframe.loc[:, self.target_column]

    def split_data(self, test_era=None):
        """
        drops non-numeric columns, splits into train and test. test being the test era
        :return: X_train, X_test, y_train, y_test
        """
        drop_columns = self.X.select_dtypes(include=["object"]).columns
        self.X_train = self.X[self.X["era"] != test_era].drop(
            drop_columns, axis=1
        )
        self.X_train = self.X_train.drop(["era"], axis=1)
        self.X_test = self.X[self.X["era"] == test_era].drop(
            drop_columns, axis=1
        )
        self.X_test = self.X_test.drop(["era"], axis=1)
        self.y_train = self.y[self.X["era"] != test_era]
        self.y_test = self.y[self.X["era"] == test_era]

    def cross_validate(self):
        """
        cross validation with groupsplitter
        :return:
        """
        splits = self.X["era"].nunique()
        group_splitter = GroupShuffleSplit(
            n_splits=splits, train_size=0.7, random_state=42
        )
        gs_iterator = group_splitter.split(self.X, self.y, groups=self.X.era)
        drop_columns = self.X.select_dtypes(include=["object"]).columns
        scores = []
        while (indices := next(gs_iterator, None)) is not None:
            self.X_train = self.X.loc[indices[0]].drop(drop_columns, axis=1)
            self.y_train = self.y.loc[indices[0]]
            self.X_test = self.X.loc[indices[1]].drop(drop_columns, axis=1)
            self.y_test = self.y.loc[indices[1]]
            self.scale_data()
            self.model.fit(self.X_train, self.y_train)
            predicted_dataframe = pd.concat(
                [self.X.loc[indices[1]], self.y.loc[indices[1]]], axis=1
            )
            predicted_dataframe["prediction"] = self.model.predict(self.X_test)
            adjusted_predicted_dataframe = self.adjust(predicted_dataframe)
            score_of_prediction = self.score(adjusted_predicted_dataframe)
            scores.append(score_of_prediction[1])
        return sum(scores) / len(scores)

    def cross_validate_time_series_classification(self):
        """
        cross validation with time series split
        :return:
        """
        splits = sorted(self.X["era"].unique())
        max_training_size = len(splits) - 1
        length_initial_training = 3
        drop_columns = self.X.select_dtypes(include=["object"]).columns
        scores = []
        for i in range(max_training_size - length_initial_training):
            training_era_bottom = splits[length_initial_training - 3]
            training_era_top = splits[length_initial_training]
            length_initial_training += 1
            test_era = splits[length_initial_training]

            self.X_train = self.X.loc[self.X["era"] <= training_era_top].drop(
                drop_columns, axis=1
            )
            self.y_train = self.y.loc[self.X["era"] <= training_era_top]
            self.y_train = self.y_train.loc[self.X_train["era"] >= training_era_bottom]
            self.X_train = self.X_train.loc[self.X_train["era"] >= training_era_bottom]

            self.X_test = self.X.loc[self.X["era"] == test_era].drop(
                drop_columns, axis=1
            )
            self.y_test = self.y.loc[self.X["era"] == test_era]
            self.scale_data()
            self.model.fit(self.X_train, self.y_train)
            scores.append(accuracy_score(self.y_test, self.model.predict(self.X_test)))
        return sum(scores) / len(scores)

    def cross_validate_time_series_accuracy(self):
        """
        cross validation with time series split
        :return:
        """
        splits = sorted(self.X["era"].unique())
        max_training_size = len(splits) - 1
        length_initial_training = 3
        drop_columns = self.X.select_dtypes(include=["object"]).columns
        scores = []
        for i in range(max_training_size - length_initial_training):
            training_era_bottom = splits[length_initial_training - 3]
            training_era_top = splits[length_initial_training]
            length_initial_training += 1
            test_era = splits[length_initial_training]

            self.X_train = self.X.loc[self.X["era"] <= training_era_top].drop(
                drop_columns, axis=1
            )
            self.y_train = self.y.loc[self.X["era"] <= training_era_top]
            self.y_train = self.y_train.loc[self.X_train["era"] >= training_era_bottom]
            self.X_train = self.X_train.loc[self.X_train["era"] >= training_era_bottom]

            self.X_test = self.X.loc[self.X["era"] == test_era].drop(
                drop_columns, axis=1
            )
            self.y_test = self.y.loc[self.X["era"] == test_era]
            self.scale_data()
            self.model.fit(self.X_train, self.y_train)
            scores.append(mean_squared_error(self.y_test, self.model.predict(self.X_test), squared=False))

        return sum(scores) / len(scores)

    def cross_validate_time_series_score(self):
        """
        cross validation with time series split
        :return:
        """
        splits = sorted(self.X["era"].unique())
        max_training_size = len(splits) - 1
        length_initial_training = 3
        drop_columns = self.X.select_dtypes(include=["object"]).columns
        scores = []
        for i in range(max_training_size - length_initial_training):
            training_era_bottom = splits[length_initial_training - 3]
            training_era_top = splits[length_initial_training]
            length_initial_training += 1
            test_era = splits[length_initial_training]

            self.X_train = self.X.loc[self.X["era"] <= training_era_top].drop(
                drop_columns, axis=1
            )
            self.y_train = self.y.loc[self.X["era"] <= training_era_top]
            self.y_train = self.y_train.loc[self.X_train["era"] >= training_era_bottom]
            self.X_train = self.X_train.loc[self.X_train["era"] >= training_era_bottom]

            self.X_test = self.X.loc[self.X["era"] == test_era].drop(
                drop_columns, axis=1
            )
            self.y_test = self.y.loc[self.X["era"] == test_era]
            self.scale_data()
            self.model.fit(self.X_train, self.y_train)
            predicted_dataframe = pd.concat(
                [
                    self.X.loc[self.X["era"] == test_era],
                    self.y.loc[self.X["era"] == test_era],
                ],
                axis=1,
            )
            predicted_dataframe["prediction"] = self.model.predict(self.X_test)
            adjusted_predicted_dataframe = self.adjust(predicted_dataframe)
            score_of_prediction = self.score(adjusted_predicted_dataframe)
            scores.append(score_of_prediction[1])
        return sum(scores) / len(scores)

    def scale_data(self):
        """
        identifies which columns are numeric and scales them
        :return:
        """
        transform_columns = self.X_train.select_dtypes(
            exclude=["object"]
        ).columns
        self.column_transformer = make_column_transformer(
            (
                MinMaxScaler(),
                transform_columns,
            ),
            remainder="passthrough",
        )
        self.X_train = self.column_transformer.fit_transform(self.X_train)
        self.X_test = self.column_transformer.transform(self.X_test)

    def save_trained_model(self):
        """
        save trained model
        :return:
        """
        filename = f"../models/trained_models/{type(self.model).__name__}.sav"
        pickle.dump(self.model, open(filename, "wb"))

    def load_trained_model(self, model_name):

        self.model = pickle.load(open(model_name, "rb"))
        return self.model

    def model_selection(self, model_type):
        """
        select model
        :param model_type:
        :return:
        """
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(random_state=42)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(random_state=42)
        elif model_type == "gradient_boosting_classifier":
            self.model = GradientBoostingClassifier(random_state=42,
                                                    learning_rate=0.010105433442925346,
                                                    max_depth=3,
                                                    n_estimators=117)
        elif model_type == "ridge":
            self.model = Ridge(random_state=42)
        elif model_type == "lasso":
            self.model = Lasso(random_state=42)
        elif model_type == "lgbm_model_2":
            self.model = LGBMRegressor(random_state=42,
                                        learning_rate=0.16718178232352315,
                                        max_depth=10,
                                        n_estimators=959
                                       )
        elif model_type == "lgbm_model_3":
            self.model = LGBMRegressor(random_state=42,
                                        learning_rate=0.17732332270120013,
                                        max_depth=10,
                                        n_estimators=966
                                       )
        elif model_type == "lgbm_classifier":
            self.model = LGBMClassifier(random_state=42,
                                        learning_rate=0.4614468946579767,
                                        max_depth=4,
                                        n_estimators=100)
        elif model_type == "xgboost_model_2":
            self.model = XGBRegressor(random_state=42,
                                       learning_rate=0.11471176253516005,
                                       max_depth=10,
                                       n_estimators=969)

        elif model_type == "xgboost_model_3":
            self.model = XGBRegressor(random_state=42,
                                       learning_rate=0.10514508380628737,
                                       max_depth=10,
                                       n_estimators=905)
        elif model_type == "xgboost_classifier":
            self.model = XGBClassifier(random_state=42,
                                       learning_rate=0.4930947860812882,
                                       max_depth=8,
                                       n_estimators=226)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError("model type not found")

    def evaluation_selection(self, evaluation_type):
        """
        select evaluation
        :param evaluation_type:
        :return:
        """
        if evaluation_type == "rmse":
            evaluation = "neg_root_mean_squared_error"
        elif evaluation_type == "mae":
            evaluation = "neg_mean_absolute_error"
        elif evaluation_type == "mape":
            evaluation = "neg_mean_absolute_percentage_error"
        elif evaluation_type == "r2":
            evaluation = "r2"
        return evaluation


def optuna_model_1(eras):
    training_dataframes = []
    for era in range(eras - 8, eras):
        training_dataframes.append(
            pd.read_csv(
                f"../../data_collection/data/processed_data/model_1_data_{era}.csv"
            )
        )

    dataframe = pd.concat(training_dataframes)

    features = [
        "overall_total_bond",
        "overall_proportional_bond",
        "nominator_count",
        "elected_previous_era",
        "era",
    ]
    target = "elected_current_era"

    return dataframe, features, target


def optuna_model_2(eras):
    training_dataframes = []
    for era in range(eras - 8, eras):
        training_dataframes.append(
            pd.read_csv(
                f"../../data_collection/data/processed_data/model_2_data_{era}.csv"
            )
        )

    dataframe = pd.concat(training_dataframes)

    features = [
        "validator",
        "proportional_bond",
        "total_bond",
        "validator_frequency_current_era",
        "probability_of_selection",
        "era",
    ]
    target = "solution_bond"

    return dataframe, features, target


def optuna_model_3(eras):
    training_dataframes = []
    for era in range(eras - 8, eras):
        training_dataframes.append(
            pd.read_csv(
                f"../../data_collection/data/processed_data/model_3_data_{era}.csv"
            )
        )

    dataframe = pd.concat(training_dataframes)

    features = [
        "nominator",
        "validator",
        "proportional_bond",
        "total_bond",
        "overall_total_bond",
        "overall_proportional_bond",
        "era",
        "number_of_validators",
        "validator_frequency_current_era",
        "average_proportional_bond",
        "average_total_bond",
        "nominator_index",
        "validator_index",
        "nominator_centrality",
        "validator_centrality",
        "probability_of_selection",
        "expected_sum_stake",
    ]

    target = "solution_bond"
    return dataframe, features, target


if __name__ == "__main__":

    type = "model_3"
    eras = 980
    name= "xgboost_model_3"

    dataframe = None
    features = None
    target = None

    if type == "model_1":
        dataframe, features, target = optuna_model_1(eras)
    elif type == "model_2":
        dataframe, features, target = optuna_model_2(eras)
    elif type == "model_3":
        dataframe, features, target = optuna_model_3(eras)

    model = Model(dataframe, target, features)

    if type == "model_2" or type == "model_3":
        model.divide_target_by_total_bond()

    direction = None
    if type == "model_2":
        direction = "minimize"
    else:
        direction = "maximize"
    study = optuna.create_study(
        direction=direction,
        storage="sqlite:///db.sqlite3",
        study_name=name,
        load_if_exists=True,
    )

    objective = None
    if type == "model_1":
        objective = model.objective_model_classification_time_series
    elif type == "model_2":
        objective = model.objective_model_accuracy_time_series
    elif type == "model_3":
        objective = model.objective_score_boosting

    study.optimize(objective, n_trials=1000)
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    """

    from sklearn.linear_model import QuantileRegressor

    dataframe = pd.read_csv(
        "../../data_collection/data/ensemble_model/total_voter_preferences.csv"
    )

    dataframe.drop(["Unnamed: 0"], axis=1, inplace=True)
    features = [
        "nominator",
        "validator",
        "proportional_bond",
        "era"
    ]
    X = dataframe.loc[:, features]
    y = dataframe.loc[:, "solution_bond"]
    X_train = X[X["era"] != 994].drop(["era", "nominator", "validator"], axis=1)
    y_train = y[X["era"] != 994]
    X_test = X[X["era"] == 994].drop(["era", "nominator", "validator"], axis=1)
    y_test = y[X["era"] == 994]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # reduce X_train rows to 10000
    X_train = X_train[:10000]
    y_train = y_train[:10000]

    model = QuantileRegressor(solver="highs")
    model.fit(X_train, y_train)
    predicted_dataframe = pd.concat(
        [X[X["era"] == 994], y[X["era"] == 994]], axis=1
    )
    predicted_dataframe["prediction"] = model.predict(X_test)
    accuracy = mean_squared_error(model.predict(X_test), y_test)
    print(accuracy)
    """

    """
    dataframe2 = pd.read_csv("../../data_collection/data/model_2/df_bond_distribution_testing_0.csv")
    dataframe2.drop(["Unnamed: 0"], axis=1, inplace=True)
    # divide solution_bond by total_bond
    dataframe2["solution_bond"] = dataframe2["solution_bond"] / dataframe2["total_bond"]
    features = [
        "nominator",
        "validator",
        "proportional_bond",
        "total_bond",
        "era"
    ]
    X = dataframe2.loc[:, features]
    y = dataframe2.loc[:, "solution_bond"]
    X_train = X[X["era"] != 994].drop(["era", "nominator", "validator"], axis=1)
    y_train = y[X["era"] != 994]
    X_test = X[X["era"] == 994].drop(["era", "nominator", "validator"], axis=1)
    y_test = y[X["era"] == 994]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model2 = GradientBoostingRegressor()
    model2.fit(X_train, y_train)
    predicted_dataframe = pd.concat(
        [X[X["era"] == 994], y[X["era"] == 994]], axis=1
    )
    predicted_dataframe["prediction"] = model2.predict(X_test)
    accuracy = mean_squared_error(model2.predict(X_test), y_test)
    print(accuracy)
    # multiply prediction by total_bond
    predicted_dataframe["prediction"] = predicted_dataframe["prediction"] * dataframe2["total_bond"]
    adjusted_predicted_dataframe = Model.adjust(predicted_dataframe)
    score = Model.score(adjusted_predicted_dataframe)



    from sklearn.ensemble import VotingRegressor
    votingregressor = VotingRegressor(estimators=[("model1", model), ("model2", model2)])
    votingregressor.fit(X_train, y_train)
    predicted_dataframe = pd.concat(
        [X[X["era"] == 994], y[X["era"] == 994]], axis=1
    )
    predicted_dataframe["prediction"] = votingregressor.predict(X_test)
    accuracy = mean_squared_error(votingregressor.predict(X_test), y_test)
    print(f"accuracy: {accuracy}")
    print()

    """
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # reading the csv file
    dataframe = pd.read_csv(
        "../../data_collection/data/ensemble_model/total_voter_preferences.csv"
    )

    # fit simple linear regression model
    linear_model = ols('solution_bond ~ proportional_bond + nominator_count',
                       data=dataframe).fit()

    # display model summary
    print(linear_model.summary())

    # modify figure size
    fig = plt.figure(figsize=(14, 8))

    # creating regression plots
    fig = sm.graphics.plot_regress_exog(linear_model,
                                        'proportional_bond',
                                        fig=fig)

    plt.show()
    """
