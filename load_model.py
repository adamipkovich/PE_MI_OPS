import mlflow
import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:5000") 
run_id = "05db52571628456583b07cf2d94ae00e"

model =  mlflow.sklearn.load_model(f"runs:/{run_id}//model")

#%% működik-e?
data = pd.read_csv("./data/cars.csv", sep=";")
client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
run_data_dict = client.get_run(run_id).data.to_dictionary()
print(model.predict(data.loc[:, eval(run_data_dict["params"]["input"])]))