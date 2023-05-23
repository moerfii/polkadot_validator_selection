import argparse
import pickle

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from lightgbm.sklearn import LGBMRegressor
import pandas as pd
from src.score import ScoringTool
from src.adjustment import AdjustmentTool
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression


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
            "regressor", ["lgbm", "xgboost"] # , "randomforest", "gradientboosting"
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
            self.model = LGBMRegressor(n_estimators=trial.suggest_int("n_estimators", 100, 1000), max_depth=trial.suggest_int("max_depth", 3, 10), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5), random_state=42)
        elif model_type == "xgboost":
            self.model = XGBRegressor(n_estimators=trial.suggest_int("n_estimators", 100, 1000), max_depth=trial.suggest_int("max_depth", 3, 10), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5), random_state=42)

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

    def objective_score_boosting(self, trial):
        self.objective(trial)
        return self.cross_validate()

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
        selector = SelectKBest(f_regression, k=10)
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
        predictions = predictions['prediction']
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
        self.X_test = self.X[self.X["era"] == test_era].drop(
            drop_columns, axis=1
        )
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

    def scale_data(self):
        """
        identifies which columns are numeric and scales them
        :return:
        """
        transform_columns = self.X_train.select_dtypes(
            exclude=["object"]
        ).columns
        column_transformer = make_column_transformer(
            (
                MinMaxScaler(),
                transform_columns,
            ),
            remainder="passthrough",
        )
        self.X_train = column_transformer.fit_transform(self.X_train)
        self.X_test = column_transformer.transform(self.X_test)

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
            self.model = RandomForestRegressor()
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor()
        elif model_type == "ridge":
            self.model = Ridge()
        elif model_type == "lasso":
            self.model = Lasso()
        elif model_type == "lgbm":
            self.model = LGBMRegressor()
        elif model_type == "xgboost":
            self.model = XGBRegressor()
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    dataframe = pd.read_csv(
        "../../data_collection/data/model_2/processed_data_expected_904.csv"
    )

    features = [
        "nominator",
        "validator",
        "proportional_bond",
        "total_bond",
        "overall_total_bond",
        "overall_proportional_bond",
        "prev_min_stake",
        "prev_sum_stake",
        "prev_variance_stake",
        "era",
        "validator_count",
        "number_of_validators"
    ]
    model = Model(dataframe, "solution_bond", features)
    model.divide_target_by_total_bond()
    model.split_data(test_era=904)
    study = optuna.create_study(
        direction="maximize", storage="sqlite:///db.sqlite3"
    )
    study.optimize(model.objective_model_accuracy, n_trials=100)
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





