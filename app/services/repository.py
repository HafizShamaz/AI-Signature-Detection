import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone
from app.config import get_settings

def timestamp():
    return datetime.now(timezone.utc).isoformat()

class Repository:
    def __init__(self):
        self.db = get_settings().database_path
        Path(self.db).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS files (file_id TEXT PRIMARY KEY, storage_key TEXT, filename TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, file_id TEXT, storage_key TEXT, operation TEXT, status TEXT, created_at TEXT, updated_at TEXT, result TEXT, error TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS queue_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT, claimed INTEGER DEFAULT 0)")

    def connect(self):
        return sqlite3.connect(self.db, timeout=30)

    def save_file(self, fid, key, name):
        with self.connect() as c:
            c.execute("INSERT INTO files VALUES (?,?,?)", (fid,key,name))

    def get_file(self, fid):
        with self.connect() as c:
            r=c.execute("SELECT file_id,storage_key,filename FROM files WHERE file_id=?", (fid,)).fetchone()
        return None if not r else {"file_id":r[0],"storage_key":r[1],"filename":r[2]}

    def create_job(self, jid, fid, key, op):
        t=timestamp()
        with self.connect() as c:
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",(jid,fid,key,op,"queued",t,t,None,None))

    def update_job(self, jid, status, result=None, error=None):
        with self.connect() as c:
            c.execute("UPDATE jobs SET status=?,updated_at=?,result=?,error=? WHERE job_id=?",
                      (status,timestamp(),json.dumps(result) if result is not None else None,error,jid))

    def get_job(self, jid):
        with self.connect() as c:
            r=c.execute("SELECT job_id,status,created_at,updated_at,result,error FROM jobs WHERE job_id=?", (jid,)).fetchone()
        if not r: return None
        return {"job_id":r[0],"status":r[1],"created_at":r[2],"updated_at":r[3],
                "result":json.loads(r[4]) if r[4] else None,"error":r[5]}
