
import os
from pathlib import Path

from app.config import get_settings

_MODEL = None
_SIMILARITY_MODEL = None
_DEVICE = "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(
    os.getenv(
        "SIGNATURE_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "signature_detector.pt")
    )
)

SIMILARITY_MODEL_PATH = Path(
    os.getenv(
        "SIGNATURE_SIMILARITY_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "signature_similarity.pt")
    )
)

CONFIDENCE_THRESHOLD = float(
    os.getenv("SIGNATURE_CONFIDENCE_THRESHOLD", "0.50")
)


def load_local_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    import torch
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn
    )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Signature model not found: {MODEL_PATH}"
        )

    print(f"Loading signature model from: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=_DEVICE,
        weights_only=True
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2
    )

    model.load_state_dict(checkpoint, strict=True)
    model.to(_DEVICE)
    model.eval()

    _MODEL = model

    print("Signature detection model loaded successfully.")

    return _MODEL


def load_similarity_model():
    global _SIMILARITY_MODEL

    if _SIMILARITY_MODEL is not None:
        return _SIMILARITY_MODEL

    from similarity.model import SignatureSimilarity

    if not SIMILARITY_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Similarity model not found: {SIMILARITY_MODEL_PATH}"
        )

    print(f"Loading similarity model from: {SIMILARITY_MODEL_PATH}")

    _SIMILARITY_MODEL = SignatureSimilarity(
        checkpoint_path=SIMILARITY_MODEL_PATH
    )

    print("Signature similarity model loaded successfully.")

    return _SIMILARITY_MODEL


class Inference:

    def detect(self, path):
        s = get_settings()

        if s.inference_backend == "stub":
            raise NotImplementedError(
                "Connect your team's licensed signature detector."
            )

        if s.inference_backend == "http":
            import httpx

            headers = (
                {"Authorization": "Bearer " + s.inference_api_key}
                if s.inference_api_key else {}
            )

            with open(path, "rb") as f:
                r = httpx.post(
                    s.inference_url.rstrip("/") + "/detect",
                    headers=headers,
                    files={"file": f},
                    timeout=180,
                )

            r.raise_for_status()
            return r.json()

        if s.inference_backend == "torchvision":
            import torch
            from PIL import Image
            from torchvision.transforms.functional import to_tensor

            model = load_local_model()

            with Image.open(path) as image:
                image = image.convert("RGB")
                image_tensor = to_tensor(image).to(_DEVICE)

            with torch.inference_mode():
                prediction = model([image_tensor])[0]

            boxes = prediction["boxes"].cpu().tolist()
            scores = prediction["scores"].cpu().tolist()
            labels = prediction["labels"].cpu().tolist()

            detections = []

            for box, score, label in zip(boxes, scores, labels):
                if label != 1 or score < CONFIDENCE_THRESHOLD:
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
                "detections": detections
            }

        raise ValueError(
            f"Unknown INFERENCE_BACKEND: {s.inference_backend}"
        )

    def compare(self, path_a, path_b):
        import torch
        from PIL import Image
        from torchvision import transforms

        model = load_similarity_model()

        # Checkpoint image_size is stored as (height, width).
        preprocess = transforms.Compose([
            transforms.Resize(model.image_size),
        ])

        def prepare_image(path):
            with Image.open(path) as image:
                image = image.convert("RGB")
                image_tensor = transforms.ToTensor()(preprocess(image))
                return image_tensor.unsqueeze(0)

        image_a = prepare_image(path_a)
        image_b = prepare_image(path_b)

        embedding_a = model.get_embedding(image_a)
        embedding_b = model.get_embedding(image_b)

        result = model.compare_embeddings(
            embedding_a,
            embedding_b
        )

        return result
