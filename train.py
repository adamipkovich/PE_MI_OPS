import sklearn

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
import mlflow


if __name__ == "__main__":

    ## hova mentse a modelt
    mlflow.set_tracking_uri("http://localhost:5000") ## ez egy lokális server,
    # terminálon futtatjuk mlflow server commanddal
    mlflow.sklearn.autolog()
    ## olvassuk be az adatunkat
    data = pd.read_excel("split_data/train.xlsx")
    X = data.drop("Output (S)", axis=1)
    y = data["Output (S)"]

    ## modellezés - Valami egyszerű legyen egy döntési fa
    with mlflow.start_run() as run:

        #tanítás

        pipe = Pipeline([
            ("minmax", sklearn.preprocessing.MinMaxScaler()),
            ## Bármilyen lépés, pl dimenzió redukció, feature engineering, stb
            ("clf", DecisionTreeClassifier())
        ])

        pipe.fit(X, y)

        ## validálás
        pred = pipe.predict(X)
        accuracy = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, normalize='true')
        tn, fp, fn, tp = cm.ravel()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
        disp.plot()


        ## mentés: plusz metrikák
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)




        # mlflow.sklearn.log_model(clf, "decision_tree_model")




