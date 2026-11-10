from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.storage import Storage
from app.services.repository import Repository

router=APIRouter(prefix="/documents",tags=["documents"])
ALLOWED={".pdf",".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

@router.post("/upload")
async def upload(file: UploadFile=File(...)):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED: raise HTTPException(415,"Unsupported file type")
    data=await file.read()
    if not data: raise HTTPException(400,"Empty upload")
    if len(data)>25*1024*1024: raise HTTPException(413,"Maximum file size is 25 MB")
    saved=Storage().save(file.filename or "document",data)
    Repository().save_file(saved["file_id"],saved["storage_key"],saved["filename"])
    return {"file_id":saved["file_id"],"filename":saved["filename"]}
