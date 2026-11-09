"""End-to-end Locust load test for the Signature Detection API.

Set LOAD_TEST_IMAGE to a synthetic/non-sensitive PNG/JPG before running.
The custom JOB metric measures actual completed jobs, not only HTTP uploads.
"""

import mimetypes
import os
import time
from pathlib import Path

from gevent import sleep
from locust import HttpUser, between, events, task

IMAGE_PATH = Path(
    os.environ.get("LOAD_TEST_IMAGE", str(Path(__file__).with_name("sample.jpg")))
).expanduser()
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "1"))
JOB_TIMEOUT = float(os.environ.get("JOB_TIMEOUT_SECONDS", "90"))
CONTENT_TYPE = mimetypes.guess_type(IMAGE_PATH.name)[0] or "application/octet-stream"


class SignatureUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        if not IMAGE_PATH.is_file():
            raise FileNotFoundError(
                f"Load test image not found: {IMAGE_PATH}. "
                "Set LOAD_TEST_IMAGE to an existing synthetic/test JPG or PNG."
            )

    @staticmethod
    def record_job(started, error=None):
        # Custom Locust statistic for complete upload -> processing -> final status.
        events.request.fire(
            request_type="JOB",
            name="upload_to_final_status",
            response_time=(time.perf_counter() - started) * 1000,
            response_length=0,
            exception=error,
            context={},
        )

    @task
    def upload_and_wait(self):
        started = time.perf_counter()

        # 1. Upload an image and obtain a new job_id.
        with IMAGE_PATH.open("rb") as image:
            with self.client.post(
                "/documents",
                files={"file": (IMAGE_PATH.name, image, CONTENT_TYPE)},
                name="POST /documents",
                catch_response=True,
                timeout=30,
            ) as response:
                if response.status_code not in (200, 201, 202):
                    msg = f"Upload returned HTTP {response.status_code}"
                    response.failure(msg)
                    self.record_job(started, RuntimeError(msg))
                    return
                try:
                    job_id = response.json().get("job_id")
                except ValueError:
                    job_id = None
                if not job_id:
                    msg = "Upload response missing job_id"
                    response.failure(msg)
                    self.record_job(started, RuntimeError(msg))
                    return

        # 2. Poll PostgreSQL-backed job endpoint until completion/failure/timeout.
        while time.perf_counter() - started < JOB_TIMEOUT:
            with self.client.get(
                f"/jobs/{job_id}",
                name="GET /jobs/{job_id}",
                catch_response=True,
                timeout=15,
            ) as response:
                if response.status_code != 200:
                    msg = f"Job lookup returned HTTP {response.status_code}"
                    response.failure(msg)
                    self.record_job(started, RuntimeError(msg))
                    return
                try:
                    job = response.json()
                except ValueError:
                    msg = "Job lookup returned invalid JSON"
                    response.failure(msg)
                    self.record_job(started, RuntimeError(msg))
                    return

                status = job.get("status")
                if status == "done":
                    self.record_job(started)
                    return
                if status == "failed":
                    msg = f"Job failed: {str(job.get('error_message') or '')[:120]}"
                    self.record_job(started, RuntimeError(msg))
                    return
                if status not in ("queued", "processing", "retrying"):
                    msg = f"Unexpected job status: {status!r}"
                    response.failure(msg)
                    self.record_job(started, RuntimeError(msg))
                    return

            sleep(POLL_INTERVAL)

        self.record_job(started, TimeoutError(f"Job did not finish within {JOB_TIMEOUT}s"))
