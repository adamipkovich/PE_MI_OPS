from fastapi import FastAPI
import mlflow
from contextlib import asynccontextmanager
import pika
import time
import logging
import pandas as pd

model = None
signature = None
current_run_id = None
rabbit_connection = None
channel = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, rabbit_connection, channel
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
    credentials = pika.PlainCredentials(username="guest", password="guest")
    while rabbit_connection is None:
        try:
            rabbit_connection = pika.BlockingConnection(pika.ConnectionParameters(host = "localhost", port = 5672, credentials=credentials, heartbeat=0))
        except pika.exceptions.AMQPConnectionError:
            logging.error(f"Connection to RabbitMQ failed at localhost:5672. Retrying...")
            time.sleep(0.3)
    channel = rabbit_connection.channel()
    channel.basic_qos(prefetch_count=1)
    yield
    channel.close()
    rabbit_connection.close()
    return
# after asnyccontextmanager

app = FastAPI(lifespan=lifespan)


@app.get("/") 
async def read_root():
    """Default path. See /docs for more."""
    return "Hello World"

@app.get("/model/current")
def get_model_state():
    global current_run_id

    if current_run_id is None:
        return "No model is loaded"
    else:
        return current_run_id 

@app.get("/model/{run_id}")
def get_mlflow_model(run_id):
    global model, signature, client, current_run_id 
    model =  mlflow.sklearn.load_model(f"runs:/{run_id}//model")
    run_data_dict = client.get_run(run_id).data.to_dictionary()  
    current_run_id = run_id
    signature = eval(run_data_dict["params"]["input"])
    return signature
#%% működik-e?

@app.get("/predict/{queue}")
async def predict(queue):
    global channel, model, signature
    method_frame, header_frame, body = channel.basic_get(queue)
    data = body.decode("utf-8")
    channel.basic_ack(method_frame.delivery_tag)
    data = pd.read_json(data)
    y = model.predict(data.loc[:, signature])
    data["y_pred"] = y
    return data.to_json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("back_end:app", host="localhost", port=8000, reload=True)