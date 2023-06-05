import json


class Selector():
    def __init__(self, era, models):
        self.era = era
        self.models = models
        self.model_name = None

    def set_model_name(self, model_name):
        self.model_name = model_name


    def load_score(self):
        with open(f"../results/{self.model_name}_{self.era}_log.json", "r") as jsonfile:
            score = json.load(jsonfile)
            return score["model"], score["score_prediction"][0]


    def select_best_configuration(self):
        scores = {}
        for model in self.models:
            self.set_model_name(model)
            model, score = self.load_score()
            scores[model] = score
        print(scores)
        print(f"Best model: {max(scores, key=scores.get)}")








if __name__ == "__main__":
    eras = range(975, 990)
    models = ["lgbm", "random_forest", "xgboost"]

    for era in eras:
        selector = Selector(era, models)
        selector.select_best_configuration()



