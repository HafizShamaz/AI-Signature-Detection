
from pathlib import Path
from tempfile import TemporaryDirectory

import psycopg
import logfire

from celery.signals import worker_process_init

from botocore.exceptions import (
    EndpointConnectionError,
    ConnectTimeoutError,
    ReadTimeoutError,
    ConnectionClosedError,
)

from shared.celery_app import celery_app

from shared.database import (
    get_connection,
    get_job,
    update_job,
)

from shared.storage import download_image


# ---------------- LOGFIRE MONITORING ----------------

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
                "signature",
            ]
        ),
    )

    logfire.instrument_celery()


# ---------------- FAILURE RECOVERY ----------------

def save_failure_or_queue(job_id, error):
    try:
        update_job(
            job_id,
            "failed",
            error_message=str(error),
        )

    except psycopg.OperationalError:
        # PostgreSQL unavailable:
        # queue the failure-status update for later.
        celery_app.send_task(
            "save_failed_status",
            args=[job_id, str(error)],
        )


# ---------------- DETECTION WORKER ----------------

@celery_app.task(
    bind=True,
    name="detect_document",
    max_retries=3,
)
def detect_document(self, job_id, object_key, model_version):

    try:
        # 1. Retrieve job from Neon PostgreSQL
        job = get_job(job_id)

        if job is None:
            raise ValueError("Job not found in PostgreSQL")

        # Avoid repeating work for a completed job.
        if job["status"] == "done":
            return job["detections"]

        # 2. Update job status
        update_job(job_id, "processing")

        # 3. Check cloud object key
        if not object_key.startswith("documents/"):
            raise ValueError("Invalid document storage key")

        extension = Path(object_key).suffix.lower()

        if extension not in (".jpg", ".jpeg", ".png"):
            raise ValueError("Unsupported image extension")

        # 4. Create temporary directory
        with TemporaryDirectory() as temp_dir:

            image_path = (
                Path(temp_dir) / f"{job_id}{extension}"
            )

            # 5. Download image from Neon Object Storage
            download_image(
                object_key,
                image_path,
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    "Image download failed"
                )

            # 6. Run signature detection
            from detection.inference import detect_signatures

            detections = detect_signatures(
                            str(image_path),
                            model_version=model_version
                        )

        # Temporary directory is automatically deleted.

        # 7. Save detection results to Neon PostgreSQL
        update_job(
            job_id,
            "done",
            detections=detections,
        )

        return detections

    # ---------------- RETRYABLE ERRORS ----------------

    except (
        psycopg.OperationalError,
        ConnectionError,
        TimeoutError,
        EndpointConnectionError,
        ConnectTimeoutError,
        ReadTimeoutError,
        ConnectionClosedError,
    ) as error:

        if self.request.retries >= self.max_retries:
            save_failure_or_queue(job_id, error)
            raise

        try:
            update_job(
                job_id,
                "retrying",
                error_message=str(error),
            )
        except psycopg.Error:
            pass

        delay = min(
            10 * (2 ** self.request.retries),
            60,
        )

        raise self.retry(
            exc=error,
            countdown=delay,
        )

    # ---------------- OTHER ERRORS ----------------

    except Exception as error:
        save_failure_or_queue(job_id, error)
        raise


# ---------------- DATABASE RECOVERY TASK ----------------

@celery_app.task(
    bind=True,
    name="save_failed_status",
    max_retries=None,
)
def save_failed_status(self, job_id, error_message):

    try:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET
                    status = 'failed',
                    error_message = %s,
                    updated_at = NOW()
                WHERE job_id = %s
                  AND status NOT IN ('done', 'failed')
                """,
                (error_message, job_id),
            )

        return "Failure status update attempted"

    except psycopg.OperationalError as error:
        raise self.retry(
            exc=error,
            countdown=30,
        )
