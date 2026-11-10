
from fastapi import FastAPI, UploadFile, File, HTTPException
from shared.database import create_job, update_job, get_job
from shared.celery_app import celery_app
from shared.storage import upload_image, delete_image

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import uuid
import logfire


app = FastAPI(title="Signature Detection API")


# ---------------- LOGFIRE ----------------

logfire.configure(
    service_name="signature-api",
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

logfire.instrument_fastapi(
    app,
    request_attributes_mapper=lambda request, attributes: {}
)

logfire.instrument_celery()


# ---------------- CONFIGURATION ----------------

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


# ---------------- API ROUTES ----------------

@app.get("/")
def home():
    return {"message": "Signature Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- UPLOAD DOCUMENT ----------------

@app.post("/documents", status_code=202)
def upload_document(file: UploadFile = File()):

    # 1. Validate extension
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG files are allowed"
        )

    # 2. Validate upload size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 10 MB"
        )

    # 3. Validate actual image content
    try:
        with Image.open(file.file) as image:

            image_format = image.format

            if image_format not in ("JPEG", "PNG"):
                raise ValueError("Unsupported image format")

            if extension == ".png" and image_format != "PNG":
                raise ValueError("File extension does not match image")

            if extension in (".jpg", ".jpeg") and image_format != "JPEG":
                raise ValueError("File extension does not match image")

            image.verify()

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    finally:
        file.file.seek(0)

    # 4. Generate unique job ID
    job_id = str(uuid.uuid4())

    # 5. Generate cloud storage key
    object_key = f"documents/{job_id}{extension}"

    content_type = (
        "image/png" if image_format == "PNG"
        else "image/jpeg"
    )

    # 6. Upload image to Neon Object Storage
    try:
        upload_image(
            file.file,
            object_key,
            content_type
        )

    except Exception as error:
        print(
            "STORAGE ERROR:",
            type(error).__name__,
            flush=True
        )

        raise HTTPException(
            status_code=503,
            detail="Cloud image storage unavailable"
        )

    # 7. Create job record in Neon PostgreSQL
    try:
        create_job(
            job_id,
            object_key,
            "v1"
        )

    except Exception as error:
        print(
            "DATABASE ERROR:",
            type(error).__name__,
            flush=True
        )

        # Clean up the uploaded object if DB insert failed
        try:
            delete_image(object_key)
        except Exception:
            print("STORAGE CLEANUP FAILED", flush=True)

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )

    # 8. Submit task to Redis through Celery
    try:
        celery_app.send_task(
            "detect_document",
            args=[
                job_id,
                object_key,
                "v1"
            ],
            task_id=job_id
        )

    except Exception as error:
        print(
            "QUEUE ERROR:",
            type(error).__name__,
            flush=True
        )

        try:
            update_job(
                job_id,
                "failed",
                error_message="Task queue unavailable"
            )
        except Exception:
            print("JOB STATUS UPDATE FAILED", flush=True)

        raise HTTPException(
            status_code=503,
            detail="Task queue unavailable"
        )

    # 9. Return job ID
    return {
        "job_id": job_id,
        "status": "queued"
    }


# ---------------- GET JOB STATUS ----------------

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):

    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    return job
