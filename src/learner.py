import numpy as np
from sklearn.ensemble import AdaBoostClassifier


class Trainer:

    def __init__(self, n=32):
        self.n = n

    def features(self):
        return self.trainer.feature_importances_.argsort()[-self.n:][::-1]

    def train(self, train, test, weights):
        self.trainer = AdaBoostClassifier(
            n_estimators=self.n)
        self.trainer.fit(train, test, sample_weight=weights)
        #print self.trainer.feature_importances_

    def score(self, data):
        scores = self.trainer.predict_proba(data)
        return scores

    def predict(self, data):
        return self.trainer.predict(data)
