import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

def initialize(n=32):
	return AdaBoostClassifier(n_estimators=n)


