
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn
)

# 1. Locate checkpoint
MODEL_PATH = Path("models/signature_detector.pt")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

# 2. Use CPU
device = torch.device("cpu")
torch.set_num_threads(4)

# 3. Load checkpoint
print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# 4. Confirm checkpoint format
if not isinstance(checkpoint, dict):
    raise TypeError("Expected a state dictionary.")

if not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
    raise ValueError(
        "Checkpoint contains additional data. "
        "We need to inspect its structure before loading."
    )

print("Checkpoint tensors:", len(checkpoint))

# 5. Reconstruct Faster R-CNN MobileNetV3
print("Reconstructing Faster R-CNN...")

model = fasterrcnn_mobilenet_v3_large_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=2
)

# 6. Strictly load trained weights
print("Loading trained weights...")

model.load_state_dict(checkpoint, strict=True)

# 7. Prepare model for inference
model.to(device)
model.eval()

print("\nMODEL LOADED SUCCESSFULLY")
print("Architecture: Faster R-CNN MobileNetV3 Large FPN")
print("Device:", device)
print("Classes: Background + Signature")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Model ready for inference.")
