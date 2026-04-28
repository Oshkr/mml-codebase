"""
Contrastive loss functions for multimodal landmark retrieval.

All losses operate on un-normalised features and apply L2 normalisation
internally before computing cosine similarity logits.
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyContrastiveLoss(nn.Module):
    """
    All-pairs contrastive loss across all available modalities.

    Computes symmetric InfoNCE for every C(n, 2) pair of active modalities
    and returns the mean.  Suitable when all modalities are equally important.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, outputs: tuple, logit_scale: torch.Tensor) -> torch.Tensor:
        feats = [f for f in outputs if f is not None]
        pairs = list(itertools.combinations(range(len(feats)), 2))

        total_loss = torch.tensor(0.0, device=self.device)
        for i, j in pairs:
            f1 = F.normalize(feats[i], dim=-1)
            f2 = F.normalize(feats[j], dim=-1)
            logits = logit_scale * f1 @ f2.T
            labels = torch.arange(len(logits), dtype=torch.long, device=self.device)
            total_loss = total_loss + self.ce(logits, labels) + self.ce(logits.T, labels)

        return total_loss / (2 * len(pairs))


class ImageBindLoss(nn.Module):
    """
    Ground-centric contrastive loss.

    Computes symmetric InfoNCE only between the ground modality (index 0)
    and each other available modality.  Inspired by ImageBind's design where
    images serve as the central binding modality.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, outputs: tuple, logit_scale: torch.Tensor) -> torch.Tensor:
        feats = [f for f in outputs if f is not None]
        # Ground is always first; pair it with every other modality.
        pairs = [(0, j) for j in range(1, len(feats))]

        total_loss = torch.tensor(0.0, device=self.device)
        for i, j in pairs:
            f1 = F.normalize(feats[i], dim=-1)
            f2 = F.normalize(feats[j], dim=-1)
            logits = logit_scale * f1 @ f2.T
            labels = torch.arange(len(logits), dtype=torch.long, device=self.device)
            total_loss = total_loss + self.ce(logits, labels) + self.ce(logits.T, labels)

        return total_loss / (2 * len(pairs))
