
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "merged_signature_dataset"

filename = "dc_00104026_png_jpg.rf.3f0ac4934c7401e879bece0c07758293"

image_path = DATASET / "train" / "images" / f"{filename}.jpg"
label_path = DATASET / "train" / "labels" / f"{filename}.txt"

print("=" * 55)
print("FULL-FRAME SAMPLE INSPECTION")
print("=" * 55)

if not image_path.exists():
    print("Image not found:", image_path)
    raise SystemExit

if not label_path.exists():
    print("Label not found:", label_path)
    raise SystemExit

with Image.open(image_path) as image:
    print("Image dimensions:", image.size)
    print("Image format:", image.format)
    print("Image mode:", image.mode)

print("\nAnnotation details:")

for line_number, line in enumerate(
    label_path.read_text().splitlines(), start=1
):
    values = line.split()

    if len(values) != 5:
        print("Invalid line:", line_number, line)
        continue

    cls, x, y, w, h = map(float, values)

    print(f"\nAnnotation {line_number}")
    print("Class:", int(cls))
    print("Center X:", x)
    print("Center Y:", y)
    print("Width:", w)
    print("Height:", h)
    print("Area:", round(w * h, 5))

print("\nInspection completed.")
