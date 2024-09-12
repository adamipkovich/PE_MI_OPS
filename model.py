import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV


#%% Load data
data = pd.read_csv("./data/cars.csv", sep=";")

ind = data["Car"]
y = data["Origin"]
X = data.drop(columns=["Origin", "Car"])

hyper_params ={'criterion': ['gini', 'entropy', 'log_loss'],
                        'ccp_alpha': [0.1, 0.01, 0.001],
                        'max_depth': np.arange(2, 10, 1), }


#%% Train decision tree model.

clf = GridSearchCV(DecisionTreeClassifier(), hyper_params, cv=5)
clf.fit(X, y) # train the model
model = clf.best_estimator_
score = clf.best_score_
y_hat = clf.predict(X)

