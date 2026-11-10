
import logging

from app.services.repository import Repository
from app.services.storage import Storage
from app.services.inference import Inference

logger = logging.getLogger(__name__)


def process(message):
    repo = Repository()
    job_id = message["job_id"]

    logger.info("Starting signature detection job: %s", job_id)

    repo.update_job(job_id, "processing")

    try:
        path = Storage().local_path(message["storage_key"])

        result = Inference().detect(path)

        if "total_signatures" not in result or "detections" not in result:
            raise ValueError(
                "Detector response must include total_signatures and detections"
            )

        repo.update_job(job_id, "completed", result=result)

        logger.info(
            "Job %s completed. Signatures detected: %s",
            job_id,
            result["total_signatures"],
        )

    except Exception as e:
        logger.exception("Signature detection job %s failed", job_id)

        repo.update_job(
            job_id,
            "failed",
            error=str(e),
        )

        raise
