import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPImageEncoder(nn.Module):
    """Frozen CLIP vision encoder that outputs visual projection features."""

    def __init__(self, model_name: str, freeze: bool = True, cache_dir: str = None):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.output_dim = self.clip_model.config.projection_dim

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.clip_model.vision_model(x).pooler_output
        return self.clip_model.visual_projection(pooled)


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder that outputs text projection features."""

    def __init__(self, model_name: str, freeze: bool = True, cache_dir: str = None):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.output_dim = self.clip_model.config.text_config.hidden_size

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, text_inputs: dict) -> torch.Tensor:
        pooled = self.clip_model.text_model(**text_inputs).pooler_output
        return self.clip_model.text_projection(pooled)
