import mlflow

import pandas as pd

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")

    runs = mlflow.search_runs()#filter_string="metrics.tn > 0.91 and metrics.tp > 0.91"
    run_id = runs["run_id"].iloc[0]

    ##get previous model
    client = mlflow.tracking.MlflowClient()
    model_name = "decision_tree_detect"
    try:
        model_metadata = client.get_latest_versions(model_name, stages=["None"])
    except mlflow.exceptions.RestException:
        client.create_registered_model(model_name)
        model_metadata = client.get_latest_versions(model_name, stages=["None"])

    if any(model_metadata):

        ### bármilyen választás logika elfogadható ... jobb metrikák stb

        # client.delete_model_version(
        #     name=model_name, version=1
        # )

        result = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )

        pass
    else:
        print(f"No model found, registering latest run as {model_name}")
        result = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )



    # alias = "best_model"
    #
    # try:
    #     prev_model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    # except mlflow.exceptions.RestException:
    #
    #
    #
    #     ## generate new registered model, with alias
    # except:
    #     print("Something went wrong")
    #
