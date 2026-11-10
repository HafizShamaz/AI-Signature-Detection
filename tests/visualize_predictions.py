
from pathlib import Path
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]

IMAGE_NAME = "fd_napptilus_00033.jpg"

IMAGE_PATH = ROOT / "merged_signature_dataset/test/images" / IMAGE_NAME
LABEL_PATH = ROOT / "merged_signature_dataset/test/labels" / f"{Path(IMAGE_NAME).stem}.txt"
PREDICTIONS_PATH = ROOT / "reports/evaluation/predictions.csv"

OUTPUT_DIR = ROOT / "reports/diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

height, width = image.shape[:2]

# Draw ground-truth boxes in GREEN
with open(LABEL_PATH, "r") as f:
    for line in f:
        values = line.strip().split()

        if len(values) != 5:
            continue

        _, xc, yc, bw, bh = map(float, values)

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, "GROUND TRUTH",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )

# Draw predictions in RED
df = pd.read_csv(PREDICTIONS_PATH)

predictions = df[
    (df["image"] == IMAGE_NAME) &
    (df["confidence"] >= 0.50)
]

for _, row in predictions.iterrows():
    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    confidence = float(row["confidence"])

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(
        image,
        f"PRED {confidence:.3f}",
        (x1, min(height - 5, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

output_path = OUTPUT_DIR / "fd_napptilus_00033_diagnostic.jpg"

cv2.imwrite(str(output_path), image)

print("Diagnostic image generated successfully:")
print(output_path)
print("Image dimensions:", width, "x", height)
print("Ground-truth annotations: 2")
print("Predictions above 0.50:", len(predictions))

