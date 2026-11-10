
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.inference import Inference

image_a = Path(
    r"C:\Users\HP ZBOOK\Desktop\signature-ai-project\merged_signature_dataset\test\images\dc_003_01_PNG_jpg.rf.fedf4ed9793cc8b779fa7de425293c6c.jpg"
)

image_b = Path(
    r"C:\Users\HP ZBOOK\Desktop\signature-ai-project\merged_signature_dataset\test\images\dc_003_15_PNG_jpg.rf.340910f2fc6fe0248840c2397fc69b6f.jpg"
)

assert image_a.is_file(), f"Image A not found: {image_a}"
assert image_b.is_file(), f"Image B not found: {image_b}"

print("Comparing signature images...")

result = Inference().compare(image_a, image_b)

print("\n========== COMPARISON RESULT ==========")
print("Similarity score:", result["similarity_score"])
print("Threshold:", result["threshold"])
print("Decision:", result["decision"])
print("=======================================")
