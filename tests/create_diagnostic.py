import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw
from app.services.inference import Inference
from pathlib import Path

image_path = Path("tests/docker_uploaded_sample.jpg")
output_path = Path("tests/signature_diagnostic.jpg")

image = Image.open(image_path).convert("RGB")
result = Inference().detect(str(image_path))

draw = ImageDraw.Draw(image)

for detection in result["detections"]:
    box = detection["bbox"]
    coordinates = (
        box["x1"], box["y1"],
        box["x2"], box["y2"]
    )

    draw.rectangle(coordinates, outline="red", width=4)

    draw.text(
        (10, 10),
        f"Confidence: {detection['confidence']}",
        fill="red"
    )

image.save(output_path, quality=90)

print("Diagnostic image:", output_path.resolve())
print("Total detections:", result["total_signatures"])
print("Bounding boxes:", result["detections"])
