from mlflow.tracking import MlflowClient
def get_experiment_id(name : str, client : MlflowClient) -> str:
    """Returns experiment id of a name in a given tracking server."""
    try:
        client.create_experiment(name)
    except Exception as e:
                pass   
    experiment_id = client.get_experiment_by_name(name).experiment_id

    return experiment_id


def pull_from_rabbit():
       pass

def push_to_rabbit():
       pass