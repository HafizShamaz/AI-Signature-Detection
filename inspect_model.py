
import torch
from pathlib import Path

MODEL_PATH = Path("models/signature_detector.pt")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load checkpoint safely.
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

print("\n========== CHECKPOINT INFORMATION ==========")
print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print("-", key)

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint)
    )

    if isinstance(state_dict, dict):
        print("\n========== MODEL PARAMETERS ==========")

        for key, value in list(state_dict.items())[:20]:
            if torch.is_tensor(value):
                print(key, tuple(value.shape))
            else:
                print(key, type(value))

else:
    print("Checkpoint is not a dictionary.")
