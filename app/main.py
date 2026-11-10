from fastapi import FastAPI
from app.routes import upload,jobs,predictions
from app.config import get_settings

s=get_settings()
app=FastAPI(title=s.app_name,version="0.1.0")
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(predictions.router)

@app.get("/")
def root(): return {"name":s.app_name,"docs":"/docs","health":"/health"}

@app.get("/health")
def health(): return {"status":"ok","environment":s.environment}
