import hmac,json,hashlib
from fastapi import FastAPI,Request,HTTPException
from app.config import get_settings
from mlops.trigger_training import submit

app=FastAPI(title="Hugging Face Dataset Webhook")
_seen=set()  # Replace with persistent idempotency storage before production.

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/webhooks/huggingface",status_code=202)
async def receive(request:Request):
    s=get_settings(); raw=await request.body()
    secret=request.headers.get("X-Webhook-Secret","")
    if not s.hf_webhook_secret or not hmac.compare_digest(secret,s.hf_webhook_secret):
        raise HTTPException(401,"Invalid webhook secret")
    try: payload=json.loads(raw)
    except ValueError: raise HTTPException(400,"Invalid JSON")
    repo=payload.get("repo",{})
    repo_id=repo.get("name") or repo.get("id") or payload.get("repo_id")
    if s.hf_allowed_repo and repo_id!=s.hf_allowed_repo: raise HTTPException(403,"Repository not allowed")
    delivery=request.headers.get("X-Webhook-Id") or hashlib.sha256(raw).hexdigest()
    if delivery in _seen: return {"status":"duplicate_ignored"}
    # For production, enqueue this event and return immediately; do not submit long work inline.
    result=submit()
    _seen.add(delivery)
    return {"status":"submitted","delivery_id":delivery,"job":result}
