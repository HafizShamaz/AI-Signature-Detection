import os,mlflow

def main():
    uri=os.getenv("MLFLOW_TRACKING_URI")
    if uri: mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("signature-detection")
    with mlflow.start_run():
        mlflow.log_param("framework","REPLACE_WITH_APPROVED_NON_ULTRALYTICS_FRAMEWORK")
        mlflow.log_param("dataset",os.getenv("HF_DATASET_REPO",""))
        raise RuntimeError("Training code is not connected. Implement dataset loading, model training, validation, real metric logging and artifact logging.")

if __name__=="__main__": main()
