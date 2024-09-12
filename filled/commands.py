import requests
import pandas as pd
import pika

def post_data(data, queue_name, host = "localhost", port = 5672, user = "guest", password = "guest"):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=pika.PlainCredentials(user, password)))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_publish(exchange='', routing_key=queue_name, body=data.encode('utf-8'))
    connection.close()

if __name__ == "__main__":
    url = "http://localhost:8000"
    run_id = "781fd38f7a0b4ef0a7e562b11eacf8e2"
    resp = requests.get(url + "/model/" + run_id)    
    print(resp)

    data = pd.read_csv("./data/cars.csv", sep=";")
    post_data(data.to_json(), "cars")
    resp = requests.get(url + "/predict/cars" )
    
    ## send this to the frontend.
    
    print(resp.content)