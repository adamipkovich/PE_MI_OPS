import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    ## keressünk jó runokat

    ## get the latest run

    runs = mlflow.search_runs()#filter_string="metrics.tn > 0.91 and metrics.tp > 0.91"
    run_id = runs["run_id"].iloc[0]

    ## predikáljunk
    data = pd.read_excel("split_data/test.xlsx")
    X = data.drop("Output (S)", axis=1)
    y = data["Output (S)"]


    ##adjuk hozzá a runhoz a validálás eredményét
    with mlflow.start_run(run_id=run_id) as run:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred)
        mlflow.log_metric("test_accuracy", accuracy)

        cm = confusion_matrix(y, pred, normalize='true')
        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metric("test_tn", tn)
        mlflow.log_metric("test_fp", fp)
        mlflow.log_metric("test_fn", fn)
        mlflow.log_metric("test_tp", tp)

