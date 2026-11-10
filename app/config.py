
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "Signature AI"
    environment: str = "development"
    api_key: str = "change-this-local-key"

    storage_backend: str = "local"
    local_data_dir: str = "./data"
    database_path: str = "./data/jobs.sqlite3"

    queue_backend: str = "sqlite"

    azure_storage_connection_string: str = ""
    azure_storage_container: str = "documents"

    azure_service_bus_connection_string: str = ""
    azure_service_bus_queue: str = "signature-jobs"

    inference_backend: str = "stub"
    inference_url: str = ""
    inference_api_key: str = ""

    hf_webhook_secret: str = ""
    hf_allowed_repo: str = ""

    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_ml_workspace: str = ""
    azure_ml_job_yaml: str = "./endpoints/training-job.yml"

    min_map50: float = 0.80
    min_precision: float = 0.80


@lru_cache
def get_settings():
    return Settings()
