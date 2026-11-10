
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "signature_similarity.pt"
)


class SiameseResNet18(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.m = resnet18(weights=None)
        self.m.fc = nn.Linear(512, embedding_dim)

    def forward(self, images):
        return self.m(images)


class SignatureSimilarity:

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):

        self.device = torch.device("cpu")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.model = SiameseResNet18(embedding_dim=128)

        self.model.load_state_dict(
            checkpoint["state"],
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        self.threshold = float(checkpoint["thr"])
        self.image_size = tuple(checkpoint["img_size"])

    @torch.inference_mode()
    def get_embedding(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        embedding = self.model(image_tensor)

        return F.normalize(embedding, p=2, dim=1)

    def compare_embeddings(self, embedding_a, embedding_b):

        similarity = F.cosine_similarity(
            embedding_a,
            embedding_b,
            dim=1
        )

        score = float(similarity.item())

        return {
            "similarity_score": round(score, 6),
            "threshold": self.threshold,
            "decision": (
                "similar"
                if score >= self.threshold
                else "different"
            )
        }
