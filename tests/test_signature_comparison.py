
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.inference import Inference


def test_signature_comparison():
    dataset_images = (
        ROOT / "merged_signature_dataset" / "test" / "images"
    )

    image_a = dataset_images / (
        "dc_003_01_PNG_jpg.rf.fedf4ed9793cc8b779fa7de425293c6c.jpg"
    )

    image_b = dataset_images / (
        "dc_003_15_PNG_jpg.rf.340910f2fc6fe0248840c2397fc69b6f.jpg"
    )

    if not image_a.is_file() or not image_b.is_file():
        pytest.skip("Signature dataset images are not available in this environment")

    print("Comparing signature images...")

    result = Inference().compare(image_a, image_b)

    assert "similarity_score" in result
    assert "threshold" in result
    assert result["decision"] in ("similar", "different")

    print("\n========== COMPARISON RESULT ==========")
    print("Similarity score:", result["similarity_score"])
    print("Threshold:", result["threshold"])
    print("Decision:", result["decision"])
    print("=======================================")
