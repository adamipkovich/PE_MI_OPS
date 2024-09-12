import mlflow
import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:5000")
run_id = "4351d55a1fbc445e968b8c573cf5f1be"
model =  mlflow.sklearn.load_model(f"runs:/{run_id}//model")
data = pd.read_csv("./data/cars.csv", sep=";")
client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
run_data_dict = client.get_run(run_id).data.to_dictionary()

print(model.predict(data.drop(columns=["Origin", "Car"])))

