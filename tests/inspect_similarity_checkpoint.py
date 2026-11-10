
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "signature_similarity.pt"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

state = checkpoint["state"]

print("=" * 60)
print("SIMILARITY MODEL DETAILS")
print("=" * 60)

print("Saved threshold:", checkpoint["thr"])
print("Expected image size:", checkpoint["img_size"])
print("State dictionary type:", type(state).__name__)
print("Number of layers/tensors:", len(state))

print("\nFirst 80 model parameters:")

for index, (name, value) in enumerate(state.items()):
    if index >= 80:
        break

    if isinstance(value, torch.Tensor):
        print(f"{name}: {tuple(value.shape)}")
    else:
        print(f"{name}: {type(value).__name__}")

print("\nFinal layers:")

for name, value in list(state.items())[-15:]:
    if isinstance(value, torch.Tensor):
        print(f"{name}: {tuple(value.shape)}")

print("\nInspection completed.")
