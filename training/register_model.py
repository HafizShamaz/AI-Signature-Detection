import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

def register(model_path,name="signature-detector"):
    client=MLClient(DefaultAzureCredential(),os.environ["AZURE_SUBSCRIPTION_ID"],
                    os.environ["AZURE_RESOURCE_GROUP"],os.environ["AZURE_ML_WORKSPACE"])
    return client.models.create_or_update(Model(path=model_path,name=name,type="custom_model"))
