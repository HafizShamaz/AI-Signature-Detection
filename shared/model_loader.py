
import hashlib
import os
import re
import tempfile
from pathlib import Path

from shared.storage import s3, BUCKET_NAME


CACHE_DIR = Path(tempfile.gettempdir()) / "signature-model-cache"


def calculate_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_detector_path(model_version="v1"):

    # Only use the approved model version.
    active_version = os.getenv(
        "DETECTION_MODEL_VERSION", "v1"
    )

    if model_version != active_version:
        raise ValueError(
            f"Model version {model_version} is not active"
        )

    if not re.fullmatch(r"v[0-9]+", model_version):
        raise ValueError("Invalid model version")

    expected_hash = os.getenv(
        "DETECTION_MODEL_SHA256", ""
    ).strip().lower()

    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise RuntimeError(
            "A valid DETECTION_MODEL_SHA256 is required"
        )

    object_key = (
        f"models/detection/{model_version}/detector.pt"
    )

    cache_folder = CACHE_DIR / model_version
    cache_folder.mkdir(parents=True, exist_ok=True)

    local_path = cache_folder / "detector.pt"

    # Reuse the existing download if its hash is valid.
    if local_path.is_file():
        if calculate_sha256(local_path) == expected_hash:
            return local_path

        local_path.unlink()

    print("Downloading detection model from cloud...", flush=True)

    # Use a temporary file to avoid partial cached models.
    with tempfile.NamedTemporaryFile(
        dir=cache_folder,
        prefix="detector-",
        suffix=".download",
        delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        s3.download_file(
            BUCKET_NAME,
            object_key,
            str(temp_path)
        )

        actual_hash = calculate_sha256(temp_path)

        if actual_hash != expected_hash:
            raise ValueError(
                "Downloaded model SHA256 does not match"
            )

        os.replace(temp_path, local_path)

    finally:
        temp_path.unlink(missing_ok=True)

    print("Model downloaded and verified", flush=True)

    return local_path
