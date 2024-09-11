import requests
import pandas as pd
import csv
import json
if __name__ == "__main__":
    url = "http://localhost:8000"
    run_id = "4351d55a1fbc445e968b8c573cf5f1be"
    resp = requests.get(url + "/model")
    print(resp)
    resp = requests.post(url + "/model/" + run_id)    
    print(resp)

    

    with open("./data/cars.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = [row for row in reader]

    data = json.dumps(data)
    
    #data = pd.read_csv(, sep=";")
    resp = requests.post(url + "/predict", data=data.to_json())
    print(resp.content)