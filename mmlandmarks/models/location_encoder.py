"""
GPS location encoder based on GeoCLIP.

Uses hierarchical Random Fourier Features (RFF) with the Equal Earth Projection
to encode latitude/longitude coordinates into fixed-size embeddings.

Reference: GeoCLIP (Vivanco et al., 2023) — https://arxiv.org/abs/2309.16020
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Random Fourier Feature utilities
# ---------------------------------------------------------------------------

def _sample_b(sigma: float, size: tuple) -> Tensor:
    return torch.randn(size) * sigma


@torch.jit.script
def _gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class GaussianEncoding(nn.Module):
    """Maps coordinates to 2*encoded_size dimensions via random Fourier features."""

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[int] = None,
        encoded_size: Optional[int] = None,
        b: Optional[Tensor] = None,
    ):
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError('"sigma", "input_size", and "encoded_size" are required.')
            b = _sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Provide only the "b" argument when using a pre-sampled matrix.')
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return _gaussian_encoding(v, self.b)


# ---------------------------------------------------------------------------
# Equal Earth Projection
# ---------------------------------------------------------------------------

def equal_earth_projection(coords: Tensor) -> Tensor:
    """
    Apply the Equal Earth Projection to (lat, lon) coordinates.

    Removes distortion introduced by treating geographic coordinates as
    Cartesian inputs.

    Args:
        coords: [N, 2] tensor of (latitude, longitude) in degrees.

    Returns:
        [N, 2] tensor of projected (x, y) coordinates.
    """
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796

    lat = coords[:, 0]
    lon = coords[:, 1]

    lbda = torch.deg2rad(lon)
    phi = torch.deg2rad(lat)

    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2.0) * torch.sin(phi)
    theta = torch.asin(sin_theta)

    denom = 3.0 * (A1 + 3 * A2 * theta**2 + theta**6 * (7 * A3 + 9 * A4 * theta**2))
    x = (2.0 * torch.sqrt(torch.tensor(3.0)) * lbda * torch.cos(theta)) / denom
    y = theta * (A1 + A2 * theta**2 + theta**6 * (A3 + A4 * theta**2))

    return torch.stack((x, y), dim=1)


# ---------------------------------------------------------------------------
# Hierarchical location encoder
# ---------------------------------------------------------------------------

class _LocationEncoderCapsule(nn.Module):
    """Single-scale location encoder using Gaussian RFF."""

    def __init__(self, sigma: float):
        super().__init__()
        rff = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.capsule = nn.Sequential(
            rff,
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
        )
        self.head = nn.Linear(1024, 512)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.capsule(x))


class LocationEncoder(nn.Module):
    """
    Hierarchical GPS encoder.

    Combines encoders at three geographic scales (1 km, 16 km, 256 km) and
    sums their outputs into a single 512-dimensional location embedding.

    The Equal Earth Projection is applied before encoding to handle
    latitude/longitude distortion.
    """

    output_dim: int = 512

    def __init__(
        self,
        sigma: list = None,
        freeze: bool = False,
        pretrained_path: str = None,
    ):
        super().__init__()
        if sigma is None:
            sigma = [2**0, 2**4, 2**8]  # 1 km, 16 km, 256 km

        self.sigma = sigma

        for i, s in enumerate(sigma):
            self.add_module(f"LocEnc{i}", _LocationEncoderCapsule(sigma=s))

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, location: Tensor) -> Tensor:
        """
        Args:
            location: [N, 2] tensor of (latitude, longitude) in degrees.

        Returns:
            [N, 512] location embedding.
        """
        location = location.float()
        location = equal_earth_projection(location)

        features = torch.zeros(location.shape[0], 512, device=location.device)
        for i in range(len(self.sigma)):
            features = features + self._modules[f"LocEnc{i}"](location)

        return features
