import argparse
import pickle
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from score import ScoringTool, ScoringUtility
from adjustment import AdjustmentTool
from sklearn.compose import make_column_transformer


class Model:
    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.total_bond = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
        #self.model = self.train()
        #self.save_trained_model()

    def objective(self, trial):
        model_type = trial.suggest_categorical("regressor", ["ridge", "lasso"])
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
            self.model = linear_model.Ridge(alpha=trial.suggest_float("alpha", 0.01, 1.0))
        elif model_type == "lasso":
            self.model = linear_model.Lasso(alpha=trial.suggest_float("alpha", 0.01, 1.0))

    def objective_model_accuracy(self, trial):
        self.objective(trial)
        evaluation = self.evaluation_selection(args.eval)
        return cross_val_score(self.model, self.X_train, self.y_train, cv=KFold(n_splits=10,
                                      shuffle=True,
                                      random_state=42), scoring=evaluation).mean()

    def objective_score_boosting(self, trial):
        self.objective(trial)
        return self.special_preprocessing_with_adjustment()

    def special_preprocessing(self):
        """
        preprocessing for boosting the score. This splits the data into a specific test era and other as training data.
        The test era is rotated and based on the 4 top eras. This should emulate a cross validation approach.
        :return:
        """
        dataframe = self.dataframe.select_dtypes(exclude=['object'])
        max_era = self.dataframe["era"].max()
        range_test_eras = range(max_era - 4, max_era)
        scores = []
        for era in range_test_eras:
            X_train = dataframe[dataframe["era"] != era]
            X_test = dataframe[dataframe["era"] == era]
            y_train = X_train[self.target_column]
            X_train = X_train.drop(self.target_column, axis=1)
            X_test = X_test.drop(self.target_column, axis=1)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            scores.append(ScoringTool.score_solution(y_pred)[0])
        return sum(scores) / len(scores)


    def special_preprocessing_with_adjustment(self):

        max_era = self.dataframe["era"].max()
        range_test_eras = range(max_era - 4, max_era) # todo: change back to 4
        scores = []
        for test_era in range_test_eras:
            print(test_era)
            X_train, X_test, y_train, y_test, total_bond = self.special_split_data(
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

            # train model
            self.model.fit(X_train, y_train)
            # predict & append to dataframe
            predicted_dataframe["prediction"] = self.model.predict(X_test)
            # adjust predictions to 100%
            adjusted_predicted_dataframe = self.adjust(predicted_dataframe)
            # score predictions
            score_of_prediction = self.score(adjusted_predicted_dataframe)
            scores.append(score_of_prediction[0])
        return sum(scores) / len(scores)

    def adjust(self, predicted_dataframe):
        """
        adjust predictions to 100%
        :param predicted_dataframe:
        :return:
        """
        adjustment_tool = AdjustmentTool()
        adjusted_predicted_dataframe = adjustment_tool.even_split_strategy(
            predicted_dataframe
        )
        return adjusted_predicted_dataframe

    @staticmethod
    def score(adjusted_predicted_dataframe):
        scorer = ScoringTool()
        predicted_dataframe = scorer.dataframe_groupby_predictions_by_validators(
            adjusted_predicted_dataframe
        )
        return scorer.score_solution(predicted_dataframe["prediction"])

    def preprocess_data(self):
        """
        preprocess data for training
        :return:
        """
        X_train, X_test, y_train, y_test = self.split_data(self.dataframe, self.target_column)
        return X_train, X_test, y_train, y_test

    def split_data(self, dataframe, target_column):
        """
        split data into train and test and drops an non-numeric columns
        :param dataframe:
        :param target_column:
        :return:
        """
        dataframe = dataframe.select_dtypes(exclude=['object'])
        X = dataframe.drop(target_column, axis=1)
        y = dataframe[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def special_split_data(self, dataframe, test_era=None):
        features = [
            "proportional_bond",
            "total_bond",
            "number_of_validators",
            "total_proportional_bond",
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
                    "number_of_validators",
                    "total_proportional_bond",
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

        dataframe = None

        return X_train, X_test, y_train, y_test, total_bond
    def save_trained_model(self):
        """
        save trained model
        :return:
        """
        filename = f"../models/trained_models/{type(self.model).__name__}.sav"
        pickle.dump(self.model, open(filename, "wb"))

    def predict(self):
        """
        predict
        :return:
        """
        return self.model.predict(self.X_test)

    def load_trained_model(self, model_name):

        self.model = pickle.load(open(model_name, "rb"))
        return self.model

    def predict_with_loaded_model(self, model_name, X_test):

            self.model = self.load_trained_model(model_name)
            return predict(self.model, X_test)

    def evaluate_with_loaded_model(self, model_name, X_test, y_test, total_bond):

            self.model = self.load_trained_model(model_name)
            return evaluate(self.predict_with_loaded_model(model_name, X_test), y_test, total_bond)



    def model_selection(self, model_type):
        """
        select model
        :param model_type:
        :return:
        """
        if model_type == "xgboost":
            model = XGBRegressor()
        elif model_type == "lightgbm":
            model = LGBMRegressor()
        return model

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
    dataframe = pd.read_csv("../../data_collection/data/model_2/df_bond_distribution_0.csv")
    dataframe.drop(["Unnamed: 0"], axis=1, inplace=True)

    model = Model(dataframe, "solution_bond")
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3", study_name="ridge_lasso")
    study.optimize(model.objective_score_boosting, n_trials=50)
    print(f"Best value: {study.best_value} (params: {study.best_params})")




