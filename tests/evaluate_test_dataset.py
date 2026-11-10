
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn
)
from torchvision.transforms.functional import to_tensor


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "merged_signature_dataset"
IMAGE_DIR = DATASET_ROOT / "test" / "images"
LABEL_DIR = DATASET_ROOT / "test" / "labels"

MODEL_PATH = PROJECT_ROOT / "models" / "signature_detector.pt"

REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"

CONFIDENCE_THRESHOLD = 0.50
AP_MIN_CONFIDENCE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IOU_THRESHOLDS = [
    round(0.50 + i * 0.05, 2)
    for i in range(10)
]


# ============================================================
# 2. LOAD TRAINED FASTER R-CNN MODEL
# ============================================================

def load_model():

    print("\nLoading trained Faster R-CNN model...")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)

    # Lower the internal threshold so AP evaluation
    # can consider lower-confidence predictions.
    setattr(model.roi_heads, "score_thresh", AP_MIN_CONFIDENCE)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Architecture: Faster R-CNN MobileNetV3 Large FPN")
    print("Device:", DEVICE)

    return model


# ============================================================
# 3. READ YOLO GROUND-TRUTH ANNOTATIONS
# ============================================================

def load_yolo_labels(label_path, image_width, image_height):

    ground_truth = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing annotation: {label_path}"
        )

    with open(label_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}, "
                    f"line {line_number}"
                )

            class_id = int(values[0])

            if class_id != 0:
                raise ValueError(
                    f"Unexpected class ID {class_id} in {label_path}"
                )

            cx, cy, width, height = map(float, values[1:])

            if not all(
                0 <= value <= 1
                for value in (cx, cy, width, height)
            ):
                raise ValueError(
                    f"Invalid normalized coordinates in {label_path}"
                )

            x1 = (cx - width / 2) * image_width
            y1 = (cy - height / 2) * image_height
            x2 = (cx + width / 2) * image_width
            y2 = (cy + height / 2) * image_height

            ground_truth.append([
                max(0.0, x1),
                max(0.0, y1),
                min(float(image_width), x2),
                min(float(image_height), y2)
            ])

    return ground_truth


# ============================================================
# 4. INTERSECTION OVER UNION
# ============================================================

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# 5. PRECISION, RECALL AND F1
# ============================================================

def calculate_fixed_threshold_metrics(
    all_predictions,
    ground_truths
):

    tp = 0
    fp = 0
    fn = 0

    image_rows = []

    for image_index, image_name in enumerate(IMAGE_NAMES):

        gt_boxes = ground_truths[image_index]

        predictions = [
            p for p in all_predictions
            if p["image_index"] == image_index
            and p["score"] >= CONFIDENCE_THRESHOLD
        ]

        matched = set()

        image_tp = 0
        image_fp = 0

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        for prediction in predictions:

            best_iou = 0.0
            best_gt_index = None

            for gt_index, gt_box in enumerate(gt_boxes):

                if gt_index in matched:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    gt_box
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= 0.50:

                matched.add(best_gt_index)
                image_tp += 1

            else:
                image_fp += 1

        image_fn = len(gt_boxes) - len(matched)

        tp += image_tp
        fp += image_fp
        fn += image_fn

        precision = (
            image_tp / (image_tp + image_fp)
            if image_tp + image_fp else 0.0
        )

        recall = (
            image_tp / (image_tp + image_fn)
            if image_tp + image_fn else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        image_rows.append({
            "image": image_name,
            "ground_truth_count": len(gt_boxes),
            "predicted_count": len(predictions),
            "true_positives": image_tp,
            "false_positives": image_fp,
            "false_negatives": image_fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6)
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }, image_rows


# ============================================================
# 6. AVERAGE PRECISION / MAP
# ============================================================

def calculate_average_precision(
    all_predictions,
    ground_truths,
    iou_threshold
):

    total_gt = sum(len(boxes) for boxes in ground_truths)

    if total_gt == 0:
        return 0.0

    predictions = sorted(
        all_predictions,
        key=lambda item: item["score"],
        reverse=True
    )

    matched = [set() for _ in ground_truths]

    true_positives = []
    false_positives = []

    for prediction in predictions:

        image_index = prediction["image_index"]
        pred_box = prediction["box"]

        gt_boxes = ground_truths[image_index]

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched[image_index]:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:

            matched[image_index].add(best_gt_index)

            true_positives.append(1)
            false_positives.append(0)

        else:

            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = 0
    cumulative_fp = 0

    recalls = []
    precisions = []

    for tp_value, fp_value in zip(
        true_positives,
        false_positives
    ):

        cumulative_tp += tp_value
        cumulative_fp += fp_value

        recall = cumulative_tp / total_gt

        precision = (
            cumulative_tp / (cumulative_tp + cumulative_fp)
        )

        recalls.append(recall)
        precisions.append(precision)

    # COCO-style 101-point interpolated AP.
    ap = 0.0

    for recall_level in [
        i / 100 for i in range(101)
    ]:

        eligible_precisions = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= recall_level
        ]

        ap += (
            max(eligible_precisions)
            if eligible_precisions else 0.0
        )

    return ap / 101


# ============================================================
# 7. MAIN EVALUATION PIPELINE
# ============================================================

def main():

    global IMAGE_NAMES

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Test image directory not found: {IMAGE_DIR}"
        )

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No test images found.")

    print("\nTest images found:", len(image_paths))

    model = load_model()

    ground_truths = []
    all_predictions = []

    IMAGE_NAMES = [p.name for p in image_paths]

    inference_times = []

    print("\nStarting full test-dataset evaluation...\n")

    for image_index, image_path in enumerate(image_paths):

        label_path = LABEL_DIR / (image_path.stem + ".txt")

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            width, height = image.size

            gt_boxes = load_yolo_labels(
                label_path,
                width,
                height
            )

            image_tensor = to_tensor(image).to(DEVICE)

        ground_truths.append(gt_boxes)

        start_time = time.perf_counter()

        with torch.inference_mode():
            prediction = model([image_tensor])[0]

        inference_times.append(
            time.perf_counter() - start_time
        )

        boxes = prediction["boxes"].cpu().tolist()
        scores = prediction["scores"].cpu().tolist()
        labels = prediction["labels"].cpu().tolist()

        for box, score, label in zip(boxes, scores, labels):

            if label != 1 or score < AP_MIN_CONFIDENCE:
                continue

            all_predictions.append({
                "image_index": image_index,
                "image": image_path.name,
                "score": float(score),
                "box": [float(v) for v in box]
            })

        if (image_index + 1) % 100 == 0 or image_index + 1 == len(image_paths):

            print(
                f"Processed {image_index + 1}/{len(image_paths)} images"
            )

    print("\nCalculating evaluation metrics...")

    # Precision / recall / F1 at confidence 0.50 and IoU 0.50.
    fixed_metrics, image_rows = calculate_fixed_threshold_metrics(
        all_predictions,
        ground_truths
    )

    # AP at each IoU threshold.
    ap_results = {}

    for threshold in IOU_THRESHOLDS:

        ap = calculate_average_precision(
            all_predictions,
            ground_truths,
            threshold
        )

        ap_results[f"AP@{threshold:.2f}"] = ap

    map_50 = ap_results["AP@0.50"]
    map_50_95 = sum(ap_results.values()) / len(ap_results)

    average_inference_time = (
        sum(inference_times) / len(inference_times)
    )

    total_ground_truth = sum(
        len(boxes) for boxes in ground_truths
    )

    summary = {
        "model": "Faster R-CNN MobileNetV3 Large FPN",
        "checkpoint": str(MODEL_PATH),
        "dataset": str(DATASET_ROOT),
        "device": str(DEVICE),
        "test_images": len(image_paths),
        "ground_truth_signatures": total_ground_truth,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold_for_precision_recall": 0.50,
        "precision": fixed_metrics["precision"],
        "recall": fixed_metrics["recall"],
        "f1_score": fixed_metrics["f1_score"],
        "true_positives": fixed_metrics["true_positives"],
        "false_positives": fixed_metrics["false_positives"],
        "false_negatives": fixed_metrics["false_negatives"],
        "mAP_50": map_50,
        "mAP_50_95": map_50_95,
        "average_inference_time_seconds": average_inference_time,
        "total_inference_time_seconds": sum(inference_times),
        "average_inference_fps": (
            1 / average_inference_time
            if average_inference_time > 0 else 0
        ),
        "average_precision_by_iou": ap_results
    }

    # Save summary JSON.
    summary_path = REPORT_DIR / "evaluation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Save per-image metrics.
    image_csv_path = REPORT_DIR / "per_image_metrics.csv"

    with open(
        image_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=image_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(image_rows)

    # Save every model prediction above AP_MIN_CONFIDENCE.
    predictions_csv_path = REPORT_DIR / "predictions.csv"

    with open(
        predictions_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for prediction in all_predictions:

            x1, y1, x2, y2 = prediction["box"]

            writer.writerow({
                "image": prediction["image"],
                "confidence": round(prediction["score"], 6),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })

    # Print final report.
    print("\n" + "=" * 55)
    print("SIGNATURE MODEL EVALUATION REPORT")
    print("=" * 55)

    print("Test images:", len(image_paths))
    print("Ground-truth signatures:", total_ground_truth)

    print("\nAt confidence 0.50 and IoU 0.50:")
    print(f"Precision: {fixed_metrics['precision']:.4f}")
    print(f"Recall:    {fixed_metrics['recall']:.4f}")
    print(f"F1-score:  {fixed_metrics['f1_score']:.4f}")

    print("\nDetection counts:")
    print("True positives:", fixed_metrics["true_positives"])
    print("False positives:", fixed_metrics["false_positives"])
    print("False negatives:", fixed_metrics["false_negatives"])

    print("\nAverage Precision:")
    print(f"mAP@0.5:     {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")

    print("\nPerformance:")
    print(f"Average inference time: {average_inference_time:.4f} seconds")
    print(f"Average inference FPS: {summary['average_inference_fps']:.2f}")

    print("\nReports saved:")
    print(summary_path)
    print(image_csv_path)
    print(predictions_csv_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
