import argparse
import pickle
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd



class Model:
    def __init__(self, dataframe, target_column, model_type):
        self.dataframe = dataframe
        self.target_column = target_column
        self.model_type = model_type
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
        model = None
        model_type = trial.suggest_categorical("regressor", ["ridge", "lasso", "randomforest", "gradientboosting"])
        """     if model_type == "xgboost":
            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                random_state=42,
            )
        elif model_type == "lightgbm":
            model = LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                random_state=42,
            )"""
        if model_type == "randomforest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                random_state=42,
            )
        elif model_type == "gradientboosting":
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
                random_state=42,
            )
        elif model_type == "ridge":
            model = linear_model.Ridge(alpha=trial.suggest_float("alpha", 0.01, 1.0))
        elif model_type == "lasso":
            model = linear_model.Lasso(alpha=trial.suggest_float("alpha", 0.01, 1.0))

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return mean_squared_error(self.y_test, y_pred)

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

    def train(self):
        """
        train model
        :return:
        """
        model = self.model_selection(self.model_type)
        model.fit(self.X_train, self.y_train)
        return model

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

    def evaluate(self):
        """
        evaluate model
        :return:
        """
        return evaluate(self.predict(), self.y_test)

    def load_trained_model(self, model_name):

        self.model = pickle.load(open(model_name, "rb"))
        return self.model

    def predict_with_loaded_model(self, model_name, X_test):

            self.model = self.load_trained_model(model_name)
            return predict(self.model, X_test)

    def evaluate_with_loaded_model(self, model_name, X_test, y_test, total_bond):

            self.model = self.load_trained_model(model_name)
            return evaluate(self.predict_with_loaded_model(model_name, X_test), y_test, total_bond)

    def optimize(self):
        """
        optimize model
        :return:
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=100)
        return study.best_params


    def optimize_with_loaded_model(self, model_name):
        """
        optimize model
        :return:
        """
        self.model = self.load_trained_model(model_name)
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=100)
        return study.best_params


    def optimize_with_loaded_model_and_save(self, model_name):
        """
        optimize model
        :return:
        """
        self.model = self.load_trained_model(model_name)
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=100)
        filename = f"../models/trained_models/{type(self.model).__name__}.sav"
        pickle.dump(self.model, open(filename, "wb"))
        return study.best_params


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
            evaluation = root_mean_squared_error
        elif evaluation_type == "mae":
            evaluation = mean_absolute_error
        elif evaluation_type == "mape":
            evaluation = mean_absolute_percentage_error
        elif evaluation_type == "r2":
            evaluation = r2_score
        return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ridge")

    args = parser.parse_args()
    dataframe = pd.read_csv("../../data_collection/data/model_2/df_bond_distribution_0.csv")

    model = Model(dataframe, "solution_bond", model_type=args.model_type)
    study = optuna.create_study(direction="minimize")
    study.optimize(model.objective, n_trials=100)




