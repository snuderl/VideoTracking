import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class Trainer:

    def __init__(self, n=32):
        self.n = n

    def features(self):
        return self.trainer.feature_importances_.argsort()[-self.n:][::-1]

    def train(self, train, test):
        self.trainer = AdaBoostClassifier(
            n_estimators=self.n,
            learning_rate=1.0,
            algorithm='SAMME.R')
        self.trainer.fit(train, test)

    def score(self, data):
        scores = self.trainer.predict_proba(data)
        return scores

    def predict(self, data):
        return self.trainer.predict(data)
