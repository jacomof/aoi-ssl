from functools import partial

from segmentation.models.vit.vit import VisionTransformer
from segmentation.models.vit.layers import (
    MemEffAttention,
    NestedTensorBlock as Block,
)


def vit_tiny(num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_small(num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        embed_dim=768,  # TODO: the kwargs do not overwrite this embed_dim
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_huge(num_register_tokens=0, **kwargs):
    return NotImplemented


def vit_giant(num_register_tokens=0, **kwargs):
    return NotImplemented


# Mapping for size identifiers to ViT constructors
VIT_SIZES = {
    "t": vit_tiny,
    "s": vit_small,
    "b": vit_base,
    "l": vit_large,
    "h": vit_huge,
    "g": vit_giant,
}


def init_vit(size, num_register_tokens=0, **kwargs):
    """
    Initialize a VisionTransformer based on size identifier.

    Args:
        size (str): Size identifier ('s', 'b', 'l', 'h', 'g').
        patch_size (int): Patch size for the ViT.
        num_register_tokens (int): Number of register tokens.
        **kwargs: Additional arguments passed to the ViT constructor.

    Returns:
        VisionTransformer instance.
    """
    if size not in VIT_SIZES:
        raise ValueError(
            f"Invalid size identifier '{size}'. Choose from {list(VIT_SIZES.keys())}."
        )
    return VIT_SIZES[size](num_register_tokens=num_register_tokens, **kwargs)


__all__ = ["init_vit", "VisionTransformer"]