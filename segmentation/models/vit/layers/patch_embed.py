# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        self.patch_grid_size = patch_grid_size
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
    

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        # Due to assertion in forward, we can assume that the image size is divisible by the patch size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def update_img_parameters(self, img_size: Union[int, Tuple[int, int]]):
        """
        Update image size parameters.
        Args:
            img_size: Image size.
        """
        self.img_size = make_2tuple(img_size)
        self.patch_grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.patches_resolution = self.patch_grid_size
        self.num_patches = self.patch_grid_size[0] * self.patch_grid_size[1]

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        self.update_img_parameters((H, W))

        # Check if the input image size is divisible by the patch size
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # input size: BxCxHxW -> output size: BxDx(H//P)x(W//P) 
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # input size: BxDx(H//P)x(W//P) -> output size: Bx[(H//P)*(W//P)]xD
        x = self.norm(x) 
        # TODO: If we don't want to flatten we shouldn't do it in the first place, so add check before flattening
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # output size:  Bx(H//P)x(W//P)xC
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops