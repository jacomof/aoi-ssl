from torch import nn
import torch
from timm.models.layers import LayerNorm2d


def make_image(x: torch.Tensor, patch_grid_size: tuple[int, int]):
    """
    Make image from patch tokens so we can downsample it.
    Args:
        x: input tensor. Shape: (B, N, C), where N is the number of tokens.
        patch_size: patch size. Tuple of integers (Px, Py).
    Returns:
        Image tensor.
    """
    print(f"Make image input shape: {x.shape}")
    print(f"Patch grid size: {patch_grid_size}")
    H = patch_grid_size[0]
    W = patch_grid_size[1]
    C = x.shape[2]

    # input shape: (B, N, C) -> output shape: (B, C, H, W), where N = H * W
    x = x.view(x.shape[0], H, W, C).permute(0, 3, 1, 2)
    return x

def make_tokens(x: torch.Tensor):
    """
    Make tokens from image so we can upsample it.
    Args:
        x: input tensor. Shape: (B, C, H, W).
        patch_size: patch size. Tuple of integers (Px, Py).
    Returns:
        Tokens tensor.
    """
    B, C, H, W = x.shape
    N = H * W
    # input shape: (B, C, H, W) -> output shape: (B, N, C), where N = H * W
    x = x.permute(0, 2, 3, 1).reshape(B, N, C)
    return x

def make_2tuple(x):
    """
    Make a 2-tuple from an integer or a tuple.
    Args:
        x: input value. Can be an integer or a tuple of integers.
    Returns:
        A 2-tuple of integers.
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

class Downsample(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        dim,
        keep_dim=False,
        patch_grid_size=None,
    ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.patch_grid_size = make_2tuple(patch_grid_size)
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )
        # Add a learnable linear projection for the cls token
        self.cls_projection = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor):
        print(f"Downsample input shape: {x.shape}")
        patch_tokens = x[:, 1:, :]  # Extract patch tokens
        cls_token = x[:, 0, :]      # Extract cls token

        # Project the cls token to the new dimension
        cls_token = self.cls_projection(cls_token)

        patch_tokens = make_image(patch_tokens, self.patch_grid_size)
        patch_tokens = self.norm(patch_tokens)
        patch_tokens = self.reduction(patch_tokens)
        print(f"Downsample output shape: {patch_tokens.shape}")
        patch_tokens = make_tokens(patch_tokens)

        # Concatenate the cls token back with the patch tokens
        x = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        return x