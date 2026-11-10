
import os
from functools import lru_cache

import torch
from PIL import Image

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn
)
from torchvision.transforms.functional import to_tensor

from shared.model_loader import get_detector_path


# Load once per worker process and model version.
@lru_cache(maxsize=2)
def load_detector(model_version="v1"):

    model_path = get_detector_path(model_version)

    print(
        f"Loading detector version {model_version}",
        flush=True
    )

    # Use only trusted, approved checkpoints.
    checkpoint = torch.load(
        str(model_path),
        map_location="cpu",
        weights_only=True
    )

    # Teammate's model expects a plain state_dict.
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected a PyTorch state_dict checkpoint"
        )

    if not all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        raise ValueError(
            "Checkpoint format does not match the "
            "expected model format"
        )

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2
    )

    # Strict loading reveals incompatible checkpoints.
    model.load_state_dict(checkpoint, strict=True)

    model.to("cpu")
    model.eval()

    print("Detection model loaded successfully", flush=True)

    return model


def detect_signatures(image_path, model_version="v1"):

    model = load_detector(model_version)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = to_tensor(image)

    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu().tolist()
    scores = prediction["scores"].cpu().tolist()
    labels = prediction["labels"].cpu().tolist()

    threshold = float(
        os.getenv("SIGNATURE_CONFIDENCE_THRESHOLD", "0.50")
    )

    detections = []

    for box, score, label in zip(
        boxes, scores, labels
    ):

        if label != 1 or score < threshold:
            continue

        x1, y1, x2, y2 = box

        detections.append({
            "label": "signature",
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2)
            }
        })

    return {
        "total_signatures": len(detections),
        "detections": detections,
        "model_version": model_version
    }
