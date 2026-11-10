
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "merged_signature_dataset/test/images"
LABEL_DIR = ROOT / "merged_signature_dataset/test/labels"
PREDICTIONS_FILE = ROOT / "reports/evaluation/predictions.csv"

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLDS = [0.50, 0.75, 0.90]

predictions = pd.read_csv(PREDICTIONS_FILE)

def load_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        _, xc, yc, bw, bh = map(float, values)

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ])

    return boxes


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


results = []

for image_path in IMAGE_DIR.iterdir():

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    import cv2

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    gt = load_ground_truth(
        LABEL_DIR / f"{image_path.stem}.txt",
        width,
        height
    )

    image_predictions = predictions[
        (predictions["image"] == image_path.name) &
        (predictions["confidence"] >= CONFIDENCE_THRESHOLD)
    ].sort_values("confidence", ascending=False)

    pred = image_predictions[
        ["x1", "y1", "x2", "y2"]
    ].values.tolist()

    matched_gt = set()
    matched_pred = set()

    iou_values = []

    for threshold in IOU_THRESHOLDS:

        used_gt = set()
        tp = 0

        for pi, p in enumerate(pred):

            best_iou = 0
            best_gt = None

            for gi, g in enumerate(gt):

                if gi in used_gt:
                    continue

                iou = calculate_iou(p, g)

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            if best_gt is not None and best_iou >= threshold:
                used_gt.add(best_gt)
                tp += 1

        if threshold == 0.50:
            matched_gt = used_gt
            matched_pred = set()

            # Reconstruct matched predictions
            used = set()
            for pi, p in enumerate(pred):
                candidates = [
                    (calculate_iou(p, g), gi)
                    for gi, g in enumerate(gt)
                    if gi not in used
                ]

                if candidates:
                    best_iou, best_gt = max(candidates)
                    if best_iou >= 0.50:
                        used.add(best_gt)
                        matched_pred.add(pi)
                        iou_values.append(best_iou)

        results.append({
            "image": image_path.name,
            "iou_threshold": threshold,
            "ground_truth": len(gt),
            "predictions": len(pred),
            "true_positives": tp,
            "false_positives": len(pred) - tp,
            "false_negatives": len(gt) - tp
        })

report = pd.DataFrame(results)

output = ROOT / "reports/diagnostics"
output.mkdir(parents=True, exist_ok=True)

report.to_csv(output / "iou_diagnostics.csv", index=False)

print("\nIOU DIAGNOSTICS")
print("============================")

for threshold in IOU_THRESHOLDS:

    d = report[report["iou_threshold"] == threshold]

    tp = d.true_positives.sum()
    fp = d.false_positives.sum()
    fn = d.false_negatives.sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(f"\nIoU threshold: {threshold}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

print("\nSaved report:")
print(output / "iou_diagnostics.csv")
