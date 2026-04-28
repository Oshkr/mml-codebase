"""
MML-CLIP: Multimodal Landmark CLIP

Aligns ground-level, satellite, text, and GPS embeddings in a shared latent space
via contrastive learning with frozen CLIP backbones and learned projection heads.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from mmlandmarks.models.encoders import CLIPImageEncoder, CLIPTextEncoder
from mmlandmarks.models.location_encoder import LocationEncoder


class MmlCLIP(nn.Module):
    """
    Four-modality contrastive model combining:
      - Ground-level images  (CLIP ViT)
      - Satellite images     (CLIP ViT, shared backbone)
      - Text descriptions    (CLIP text encoder)
      - GPS coordinates      (hierarchical location encoder)

    Each modality passes through a dedicated two-layer projection MLP before
    being L2-normalised for contrastive similarity.

    Args:
        model_name:   HuggingFace model ID for the CLIP backbone.
        freeze:       Whether to freeze the CLIP backbone weights.
        cache_dir:    Local directory for caching HuggingFace model files.
        output_dim:   Dimensionality of the shared embedding space.
        gps_sigma:    Frequency scales for the hierarchical GPS encoder.
        gps_pretrained_path: Optional path to pretrained GPS encoder weights.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",
        freeze: bool = True,
        cache_dir: str = None,
        output_dim: int = 512,
        gps_sigma: list = None,
        gps_pretrained_path: str = None,
    ):
        super().__init__()

        # --- Shared image backbone ---
        self.image_encoder = CLIPImageEncoder(model_name, freeze=freeze, cache_dir=cache_dir)
        image_dim = self.image_encoder.output_dim

        # --- Text backbone ---
        self.text_encoder = CLIPTextEncoder(model_name, freeze=freeze, cache_dir=cache_dir)
        text_dim = self.text_encoder.output_dim

        # --- GPS encoder ---
        if gps_sigma is None:
            gps_sigma = [2**0, 2**4, 2**8]
        self.location_encoder = LocationEncoder(
            sigma=gps_sigma,
            freeze=False,
            pretrained_path=gps_pretrained_path,
        )

        # --- Learnable temperature ---
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # --- Modality-specific projection heads ---
        self.ground_projector = self._make_projector(image_dim, output_dim)
        self.satellite_projector = self._make_projector(image_dim, output_dim)
        self.text_projector = self._make_projector(text_dim, output_dim)

    @staticmethod
    def _make_projector(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        ground: Optional[Tensor] = None,
        satellite: Optional[Tensor] = None,
        text: Optional[dict] = None,
        gps: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            ground:    [B, C, H, W] ground-level image batch.
            satellite: [B, C, H, W] satellite image batch.
            text:      Dict of tokenised text inputs (input_ids, attention_mask).
            gps:       [B, 2] tensor of (latitude, longitude) in degrees.

        Returns:
            Tuple of (ground_feat, satellite_feat, text_feat, gps_feat).
            Any modality passed as None returns None.
        """
        ground_feat = None
        satellite_feat = None
        text_feat = None
        gps_feat = None

        if ground is not None:
            ground_feat = self.ground_projector(self.image_encoder(ground))

        if satellite is not None:
            satellite_feat = self.satellite_projector(self.image_encoder(satellite))

        if text is not None:
            text_feat = self.text_projector(self.text_encoder(text))

        if gps is not None:
            gps_feat = self.location_encoder(gps)

        return ground_feat, satellite_feat, text_feat, gps_feat