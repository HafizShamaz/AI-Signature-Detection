from pathlib import Path
from app.config import get_settings

def submit():
    s=get_settings()
    if not all([s.azure_subscription_id,s.azure_resource_group,s.azure_ml_workspace]):
        raise RuntimeError("Configure Azure ML subscription, resource group and workspace")
    from azure.ai.ml import MLClient,load_job
    from azure.identity import DefaultAzureCredential
    client=MLClient(DefaultAzureCredential(),s.azure_subscription_id,s.azure_resource_group,s.azure_ml_workspace)
    p=Path(s.azure_ml_job_yaml)
    if not p.exists(): raise FileNotFoundError(p)
    job=load_job(source=str(p))
    result=client.jobs.create_or_update(job)
    return {"name":result.name,"status":result.status,"studio_url":result.studio_url}

if __name__=="__main__": print(submit())
