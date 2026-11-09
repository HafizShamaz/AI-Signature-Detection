from fastapi import FastAPI,UploadFile,File,HTTPException
from shared.database import create_job,update_job,get_job
from shared.celery_app import celery_app
from pathlib import Path
import uuid
from PIL import Image
import shutil

app = FastAPI(title="Signature Detection API")


UPLOAD_DIR= Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)





@app.get("/")
def home():
    return {"message": "Signature Detection API is running"}

@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/documents",status_code=202)
def uplaod_document(file: UploadFile=File()):

     # Allow only JPG and PNG files
    extension = Path(file.filename or "").suffix.lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG files are allowed"
        )

 # Generate a unique ID
    job_id = str(uuid.uuid4())

    # Save image with a unique name
    file_path = UPLOAD_DIR / (job_id + extension)

    try:
        with open(file_path, "wb") as destination:
            shutil.copyfileobj(file.file, destination)

        # Validate actual image contents
        with Image.open(file_path) as image:
            image.verify()

    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    finally:
        file.file.close()

   
    # 1. Create job record in PostgreSQL
    try:
        create_job(job_id, file_path.name, "v1")

    except Exception as error:
        print("DATABASE ERROR:", type(error).__name__, flush=True)

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )

    # 2. Submit task to Redis through Celery
    try:
        celery_app.send_task(
            "detect_document",
            args=[job_id, file_path.name, "v1"],
            task_id=job_id
        )

    except Exception as error:
        update_job(
            job_id,
            "failed",
            error_message=str(error)
        )

        raise HTTPException(
            status_code=503,
            detail="Task queue unavailable"
        )

    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):

    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    return job

