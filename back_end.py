## FastAPI
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import mlflow
import pandas as pd
import json
model = None
client = None
signature = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
    yield
    return

app = FastAPI(lifespan=lifespan)

@app.get("/") 
async def read_root():
    """Default path. See /docs for more."""
    return "Hello World"
    ## TODO:

@app.post("/model/{run_id}")
def get_mlflow_model(run_id : str):
    global model, client, signature
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model =  mlflow.sklearn.load_model(f"runs:/{run_id}//model")
    run_data_dict = client.get_run(run_id).data.to_dictionary()
    print(run_data_dict)
    signature = run_data_dict["params"]["input"]

@app.get("/model")
def get_model():
    if model is None:
        return "Model is not loaded."
    else:
        return model.__class__

@app.post("/predict")
async def predict(req : Request):
    a = await req.body()
    data = json.loads(a.decode("utf-8"))
    print(pd.json_normalize(data, max_level= 2))
    return data
    #return model.predict(data.loc[:, signature])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("back_end:app", host="localhost", port=8000, reload=True)
    