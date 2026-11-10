
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "merged_signature_dataset"

for split in ["train", "valid", "test"]:

    label_dir = DATASET / split / "labels"

    print("\n" + "=" * 50)
    print(f"{split.upper()} DATASET")
    print("=" * 50)

    if not label_dir.exists():
        print("LABEL DIRECTORY NOT FOUND:", label_dir)
        continue

    files = list(label_dir.glob("*.txt"))

    boxes = []
    full_frame = 0
    full_width = 0
    full_height = 0
    invalid = 0

    for file in files:
        for line in file.read_text().splitlines():

            values = line.split()

            if len(values) != 5:
                invalid += 1
                continue

            try:
                cls, x, y, w, h = map(float, values)
            except ValueError:
                invalid += 1
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid += 1
                continue

            boxes.append([w, h, w * h])

            if w >= 0.999 and h >= 0.999:
                full_frame += 1

            if w >= 0.999:
                full_width += 1

            if h >= 0.999:
                full_height += 1

    if boxes:
        data = np.array(boxes)

        print("Label files:", len(files))
        print("Total signatures:", len(boxes))
        print("Median width:", round(float(np.median(data[:, 0])), 4))
        print("Median height:", round(float(np.median(data[:, 1])), 4))
        print("Median area:", round(float(np.median(data[:, 2])), 5))
        print("Full-frame annotations:", full_frame)
        print("Full-width annotations:", full_width)
        print("Full-height annotations:", full_height)
        print("Invalid annotations:", invalid)

    else:
        print("No valid annotations found.")

print("\nAnnotation quality statistics completed.")
