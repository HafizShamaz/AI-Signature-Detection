import os
from celery import Celery

celery_app = Celery(
    "signature_detection",
    broker=os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0"
    ),
    include=["worker.tasks"]
)


celery_app.conf.update(
    task_default_queue = "signature_detection",
    task_serializer = "json",
    accept_content=["json"],
    task_acks_late = True,
    worker_prefetch_multiplier = 1,


     task_reject_on_worker_lost=True,

      # Redis message recovery timeout
    broker_transport_options={
        "visibility_timeout": 3600
    },
)

