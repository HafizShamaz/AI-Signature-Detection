
import sys
from pathlib import Path

# Add project root to Python import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from similarity.model import SignatureSimilarity

print("Loading signature similarity model...")

similarity = SignatureSimilarity()

print("\nMODEL LOADED SUCCESSFULLY")
print("Architecture: Siamese ResNet-18")
print("Embedding dimension: 128")
print("Image size from checkpoint:", similarity.image_size)
print("Saved threshold:", similarity.threshold)

dummy_image = torch.rand(1, 3, 128, 192)
embedding = similarity.get_embedding(dummy_image)

print("\nEmbedding shape:", tuple(embedding.shape))
print("Embedding norm:", round(embedding.norm().item(), 6))
print("\nCheckpoint integration test completed.")
