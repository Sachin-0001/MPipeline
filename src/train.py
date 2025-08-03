import os
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Sachin-0001/MPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Sachin-0001"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "74e21173621c3e789ac4d2eecefcf39e38fb6788"

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search


params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/Sachin-0001/MPipeline.mlflow")

    with mlflow.start_run():
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
        signature = infer_signature(X_train,y_train)

        param_grid = {
            'n_estimators' : [100,200],
            'max_depth' : [5,10,None],
            'min_samples_split' : [2,5],
            'min_samples_leaf' : [1,2]
        }

        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"Accuracy :{accuracy}")

        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("Best_n_estimators",grid_search.best_params_['n_estimators'])
        mlflow.log_param("Best_max_depth",grid_search.best_params_['max_depth'])
        mlflow.log_param("Best_min_samples_split",grid_search.best_params_['min_samples_split'])
        mlflow.log_param("Best_min_samples_leaf",grid_search.best_params_['min_samples_leaf'])

        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),"confusion matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # if tracking_url_type_store != "file":
        #     mlflow.sklearn.log_model(best_model,"model",signature=signature)
        # else:
        #     mlflow.sklearn.log_model(best_model,"model", signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        filename = model_path
        pickle.dump(best_model, open(filename, "wb"))

        # Log the model file as an artifact instead of using mlflow.sklearn.log_model
        mlflow.log_artifact(filename)

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train(params["data"], params["model"], params["random_state"], params["n_estimators"],params["max_depth"])
