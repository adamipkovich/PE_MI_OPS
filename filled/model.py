import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn


#%% Load data
data = pd.read_csv("./data/cars.csv", sep=";")

ind = data["Car"]
y = data["Origin"]
X = data.drop(columns=["Origin", "Car"])

hyper_params ={'criterion': ['gini', 'entropy', 'log_loss'],
                        'ccp_alpha': [0.1, 0.01, 0.001],
                        'max_depth': np.arange(2, 10, 1), }


#%% Train with gridsearch...

clf = GridSearchCV(DecisionTreeClassifier(), hyper_params, cv=5)
clf.fit(X, y) # train the model
model = clf.best_estimator_
score = clf.best_score_
y_hat = clf.predict(X)
#cm = confusion_matrix(y, y_hat, normalize='true')

#%% Save with Mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000") # To which server to upload
client = MlflowClient("http://127.0.0.1:5000") 
name = "cars" 
try:
        client.create_experiment(name)
except Exception as e:
            pass   
experiment_id = client.get_experiment_by_name(name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    run_id = mlflow.active_run().info.run_id
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("input", X.columns.to_list())
    

#%% Deployment functions -- 


print("Done!")