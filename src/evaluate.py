import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Sachin-0001/MPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Sachin-0001"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "74e21173621c3e789ac4d2eecefcf39e38fb6788"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y =data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/Sachin-0001/MPipeline.mlflow")

    # load the model from the disc
    model = pickle.load(open(model_path,"rb"))

    predictions =  model.predict(X)
    accuracy = accuracy_score(y,predictions)

    #log metrics
    mlflow.log_metric("Accuracy", accuracy)
    print("Model Accuracy", accuracy)

if __name__ == "__main__":
    evaluate(params["data"],params["model"])