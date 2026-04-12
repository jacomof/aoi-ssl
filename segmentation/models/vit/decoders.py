from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.upernet_decoder import UperNetDecoder as _UperNetDecoder


class LinearDecoder(nn.Module):
    def __init__(self, output_size, num_classes: int, embed_dim: int, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.num_classes = num_classes
        self.proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        b, n, c = patch_embeddings.shape
        hw = int(math.sqrt(n))
        x = patch_embeddings.transpose(1, 2).reshape(b, c, hw, hw)
        x = self.proj(x)
        return F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)


class FPNDecoder(nn.Module):
    def __init__(self, output_size, num_classes: int, embed_dim: int, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features
        if x.ndim == 3:
            b, n, c = x.shape
            hw = int(math.sqrt(n))
            x = x.transpose(1, 2).reshape(b, c, hw, hw)
        x = self.head(x)
        return F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)


class UperNetDecoder(nn.Module):
    def __init__(
        self,
        output_size,
        num_classes: int,
        embed_dim: int,
        fpn_inplanes=(384, 384, 384, 384),
        pool_scales=(2, 4, 8, 16),
        **kwargs,
    ):
        super().__init__()
        self.decoder = _UperNetDecoder(
            resolution=output_size,
            num_classes=num_classes,
            ppm_dim=embed_dim,
            fpn_inplanes=fpn_inplanes,
            pool_scales=pool_scales,
        )

    def forward(self, features):
        return self.decoder(features)
