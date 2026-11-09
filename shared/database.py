import os
import json
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://signature:signature_dev_password@localhost:5432/signature_db"
)


def get_connection():
    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row
    )


def create_job(job_id, filename, model_version):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs
            (job_id, status, storage_key, model_version)
            VALUES (%s, %s, %s, %s)
            """,
            (job_id, "queued", filename, model_version)
        )


def get_job(job_id):
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT job_id, status, storage_key,
                   model_version, detections, error_message
            FROM jobs
            WHERE job_id = %s
            """,
            (job_id,)
        )

        return result.fetchone()


def update_job(job_id, status, detections=None, error_message=None):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = %s,
                detections = %s::jsonb,
                error_message = %s,
                updated_at = NOW()
            WHERE job_id = %s
            """,
            (
                status,
                json.dumps(detections) if detections is not None else None,
                error_message,
                job_id
            )
        )
