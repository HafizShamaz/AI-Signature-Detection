from PIL import Image
from pathlib import Path

path = Path("tests/sample_document.jpg.jpg")

with Image.open(path) as img:
    print("IMAGE DIAGNOSTICS")
    print("-----------------")
    print("File:", path.resolve())
    print("Format:", img.format)
    print("Dimensions:", img.size)
    print("Color mode:", img.mode)
    print("File size:", round(path.stat().st_size / 1024, 2), "KB")

