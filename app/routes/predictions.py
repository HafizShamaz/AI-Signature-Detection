
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.repository import Repository
from app.services.storage import Storage
from app.services.inference import Inference

router = APIRouter(tags=["predictions"])


@router.get("/results/{job_id}")
def result(job_id: str):
    job = Repository().get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] != "completed":
        raise HTTPException(409, f"Job status: {job['status']}")

    return job["result"]


class SimilarityRequest(BaseModel):
    signature_a_file_id: str
    signature_b_file_id: str


@router.post("/similarity")
def similarity(req: SimilarityRequest):
    repo = Repository()

    a = repo.get_file(req.signature_a_file_id)
    b = repo.get_file(req.signature_b_file_id)

    if not a or not b:
        raise HTTPException(404, "Unknown file ID")

    try:
        result = Inference().compare(
            Storage().local_path(a["storage_key"]),
            Storage().local_path(b["storage_key"])
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    except Exception as e:
        raise HTTPException(500, str(e))

    return {
        **result,
        "scale": "0_to_1",
        "note": "Similarity does not prove authenticity"
    }
