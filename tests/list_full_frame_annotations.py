
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "merged_signature_dataset"

for split in ["train", "valid", "test"]:
    label_dir = DATASET / split / "labels"
    image_dir = DATASET / split / "images"

    print(f"\n{'=' * 60}")
    print(f"{split.upper()} - FULL-FRAME ANNOTATIONS")
    print("=" * 60)

    matches = []

    for label_file in label_dir.glob("*.txt"):
        for line_number, line in enumerate(
            label_file.read_text().splitlines(), start=1
        ):
            values = line.split()

            if len(values) != 5:
                continue

            try:
                cls, x, y, w, h = map(float, values)
            except ValueError:
                continue

            if w >= 0.999 and h >= 0.999:
                image_file = None

                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = image_dir / (label_file.stem + ext)
                    if candidate.exists():
                        image_file = candidate
                        break

                matches.append(
                    (label_file.name, line_number, image_file)
                )

    print("Full-frame boxes:", len(matches))

    for label_name, line_number, image_file in matches[:30]:
        print("\nLabel:", label_name)
        print("Line:", line_number)
        print("Image:", image_file if image_file else "IMAGE NOT FOUND")

    if len(matches) > 30:
        print(f"\nShowing first 30 of {len(matches)} matches.")
