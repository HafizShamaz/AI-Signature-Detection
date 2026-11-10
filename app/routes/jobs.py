from uuid import uuid4
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.repository import Repository
from app.services.queue import Queue

router=APIRouter(prefix="/jobs",tags=["jobs"])
class JobRequest(BaseModel):
    file_id:str
    operation:str="signature_detection"

@router.post("",status_code=202)
def submit(req:JobRequest):
    repo=Repository(); f=repo.get_file(req.file_id)
    if not f: raise HTTPException(404,"Unknown file_id")
    jid=str(uuid4())
    repo.create_job(jid,req.file_id,f["storage_key"],req.operation)
    try: Queue().send({"job_id":jid,"file_id":req.file_id,"storage_key":f["storage_key"],"operation":req.operation})
    except Exception as e:
        repo.update_job(jid,"failed",error=str(e))
        raise HTTPException(503,"Queue submission failed")
    return {"job_id":jid,"status":"queued"}

@router.get("/{job_id}")
def status(job_id:str):
    result=Repository().get_job(job_id)
    if not result: raise HTTPException(404,"Job not found")
    return result
