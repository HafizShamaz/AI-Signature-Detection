
from pathlib import Path
from hashlib import sha256
from dotenv import load_dotenv

load_dotenv()

from shared.storage import s3, BUCKET_NAME


model_path = Path("model_artifacts/detector.pt")

if not model_path.is_file():
    raise FileNotFoundError(
        f"Model file missing: {model_path}"
    )

object_key = "models/detection/v1/detector.pt"

with model_path.open("rb") as file:
    model_hash = sha256(file.read()).hexdigest()

s3.upload_file(
    str(model_path),
    BUCKET_NAME,
    object_key
)

print("Model uploaded successfully")
print("Storage key:", object_key)
print("SHA256:", model_hash)
