# This file includes code adapted from Meta Platforms, Inc. and affiliates:
# https://github.com/facebookresearch/dino
#
# Original code license: Apache License 2.0.
# You may obtain a copy of the license in this repository's LICENSE file.

# Modifications in this repository:
# - Added support for xFormers.
# - Reorganized and split into components.

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
