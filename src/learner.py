import numpy as np
from sklearn.ensemble import AdaBoostClassifier


class Trainer:

    def __init__(self, n=32):
        self.n = n
        self.trainer = AdaBoostClassifier(
            n_estimators=self.n)
    def features(self):
        f = np.array(self.trainer.feature_importances_.argsort()[-self.n:][::-1])
        return f 

    def train(self, train, test, weights):
        self.trainer.fit(train, test, sample_weight=weights)

    def score(self, data):
        scores = self.trainer.predict_proba(data)
        return scores

    def predict(self, data):
        return self.trainer.predict(data)
