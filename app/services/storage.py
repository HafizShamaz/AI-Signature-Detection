from pathlib import Path
from uuid import uuid4
from app.config import get_settings

class Storage:
    def __init__(self):
        self.s=get_settings()
        self.folder=Path(self.s.local_data_dir)/"uploads"
        self.folder.mkdir(parents=True,exist_ok=True)

    def save(self, filename, data):
        fid=str(uuid4())
        safe=Path(filename).name.replace("/", "_").replace("\\\\", "_")
        if self.s.storage_backend == "azure":
            from azure.storage.blob import BlobServiceClient
            client=BlobServiceClient.from_connection_string(self.s.azure_storage_connection_string)
            blob=client.get_blob_client(self.s.azure_storage_container,f"{fid}/{safe}")
            blob.upload_blob(data,overwrite=False)
            return {"file_id":fid,"storage_key":f"{fid}/{safe}","filename":safe}
        path=self.folder/f"{fid}_{safe}"
        path.write_bytes(data)
        return {"file_id":fid,"storage_key":str(path),"filename":safe}

    def local_path(self,key):
        if self.s.storage_backend == "azure":
            import tempfile
            from azure.storage.blob import BlobServiceClient
            client=BlobServiceClient.from_connection_string(self.s.azure_storage_connection_string)
            blob=client.get_blob_client(self.s.azure_storage_container,key)
            fd,path=tempfile.mkstemp(suffix=Path(key).suffix)
            import os
            with os.fdopen(fd,"wb") as f: f.write(blob.download_blob().readall())
            return path
        p=Path(key).resolve()
        if not p.is_relative_to(self.folder.resolve()) or not p.exists():
            raise FileNotFoundError(key)
        return str(p)
