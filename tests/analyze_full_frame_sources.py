
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "merged_signature_dataset" / "train" / "labels"

sources = Counter()
affected_images = []

for label_file in LABEL_DIR.glob("*.txt"):
    found = False

    for line in label_file.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        try:
            cls, x, y, w, h = map(float, values)
        except ValueError:
            continue

        if w >= 0.999 and h >= 0.999:
            found = True
            break

    if found:
        affected_images.append(label_file.stem)

        # Roboflow filenames often contain source identifiers.
        source = label_file.stem.split("_")[0]
        sources[source] += 1

print("Affected images:", len(affected_images))
print("\nMost common filename prefixes:")

for source, count in sources.most_common(20):
    print(f"{source}: {count}")

print("\nFirst 20 affected image names:")

for name in affected_images[:20]:
    print(name)

output = ROOT / "reports" / "full_frame_annotations.txt"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(affected_images))

print("\nComplete image list saved to:", output)
