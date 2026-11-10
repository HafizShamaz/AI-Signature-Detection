
from pathlib import Path

import pytest

from app.config import get_settings
from app.services.inference import Inference

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "tests" / "sample_document.jpg.jpg"


def test_local_signature_detection():
    settings = get_settings()

    # CI uses the stub backend, which cannot perform real detection.
    if settings.inference_backend == "stub":
        pytest.skip("Real signature detection is not available with stub backend")

    # Real inference requires the test image.
    if not IMAGE_PATH.is_file():
        pytest.skip("Test image is not available in this environment")

    result = Inference().detect(str(IMAGE_PATH))

    print("\n========== SIGNATURE DETECTION ==========")
    print("Total signatures:", result["total_signatures"])

    for index, detection in enumerate(result["detections"], start=1):
        print(f"\nSignature {index}")
        print("Confidence:", detection["confidence"])
        print("Bounding box:", detection["bbox"])

    print("\n==========================================")

    assert "total_signatures" in result
    assert "detections" in result
    assert isinstance(result["total_signatures"], int)
    assert isinstance(result["detections"], list)

    for detection in result["detections"]:
        assert detection["label"] == "signature"
        assert 0 <= detection["confidence"] <= 1
        assert "bbox" in detection
