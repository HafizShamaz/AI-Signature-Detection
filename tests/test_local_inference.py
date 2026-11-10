
from pathlib import Path

from app.services.inference import Inference

# Existing test image
IMAGE_PATH = Path("tests/sample_document.jpg.jpg")

def test_local_signature_detection():
    assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"

    # Run local Faster R-CNN inference
    result = Inference().detect(str(IMAGE_PATH))

    # Display results
    print("\n========== SIGNATURE DETECTION ==========")
    print("Total signatures:", result["total_signatures"])

    for index, detection in enumerate(result["detections"], start=1):
        print(f"\nSignature {index}")
        print("Confidence:", detection["confidence"])
        print("Bounding box:", detection["bbox"])

    print("\n==========================================")

    # Validate response structure
    assert "total_signatures" in result
    assert "detections" in result
    assert isinstance(result["total_signatures"], int)
    assert isinstance(result["detections"], list)

    for detection in result["detections"]:
        assert detection["label"] == "signature"
        assert 0 <= detection["confidence"] <= 1
        assert "bbox" in detection
