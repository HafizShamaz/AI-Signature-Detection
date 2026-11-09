
from pathlib import Path
import psycopg

from shared.celery_app import celery_app
from shared.database import get_connection, get_job, update_job

import logfire
from celery.signals import worker_process_init

PROJECT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_DIR / "uploads"




@worker_process_init.connect(weak=False)
def setup_worker_logfire(**kwargs):

    logfire.configure(
        service_name="signature-worker",
        send_to_logfire="if-token-present",
        console=False,
        scrubbing=logfire.ScrubbingOptions(
            extra_patterns=[
                "args",
                "kwargs",
                "filename",
                "storage_key",
                "error_message",
                "signature"
            ]
        )
    )

    logfire.instrument_celery()


def save_failure_or_queue(job_id, error):
   
    try:
        update_job(
            job_id,
            "failed",
            error_message=str(error)
        )

    except psycopg.OperationalError:
        celery_app.send_task(
            "save_failed_status",
            args=[job_id, str(error)]
        )


@celery_app.task(
    bind=True,
    name="detect_document",
    max_retries=3
)
def detect_document(self, job_id, filename, model_version):

    try:
        job = get_job(job_id)

        if job is None:
            raise ValueError("Job not found in PostgreSQL")

        if job["status"] == "done":
            return job["detections"]

        update_job(job_id, "processing")

        image_path = UPLOAD_DIR / filename

        if not image_path.is_file():
            raise FileNotFoundError(
                "Uploaded image not found"
            )

        from detection.inference import detect_signatures

        detections = detect_signatures(str(image_path))

        update_job(
            job_id,
            "done",
            detections=detections
        )

        return detections

    # Temporary errors: retry up to 3 times
    except (
        psycopg.OperationalError,
        ConnectionError,
        TimeoutError
    ) as error:

        if self.request.retries >= self.max_retries:
            save_failure_or_queue(job_id, error)
            raise

        try:
            update_job(
                job_id,
                "retrying",
                error_message=str(error)
            )
        except psycopg.Error:
            pass

        delay = min(
            10 * (2 ** self.request.retries),
            60
        )

        raise self.retry(
            exc=error,
            countdown=delay
        )

    # Other errors: fail immediately
    except Exception as error:
        save_failure_or_queue(job_id, error)
        raise


@celery_app.task(
    bind=True,
    name="save_failed_status",
    max_retries=None
)
def save_failed_status(self, job_id, error_message):

    try:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error_message = %s,
                    updated_at = NOW()
                WHERE job_id = %s
                  AND status NOT IN ('done', 'failed')
                """,
                (error_message, job_id)
            )

        return "Failure status update attempted"

    except psycopg.OperationalError as error:
        raise self.retry(
            exc=error,
            countdown=30
        )



@celery_app.task(name="test_parallel")
def test_parallel(seconds=10):
    import time
    import socket

    worker = socket.gethostname()

    print(f"START: {worker}", flush=True)

    time.sleep(seconds)

    print(f"FINISH: {worker}", flush=True)

    return "Completed"