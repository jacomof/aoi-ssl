import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class CopyFirstChannelToThird(ImageOnlyTransform):
    """Convert 2-channel image to 3-channel by copying the first channel to the third."""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # img shape: (H, W, 2)
        if img.shape[-1] != 2:
            raise ValueError(f"Expected 2-channel image, got shape {img.shape}")
        h, w, _ = img.shape
        out = np.zeros((h, w, 3), dtype=img.dtype)
        out[..., 0] = img[..., 0]
        out[..., 1] = img[..., 1]
        out[..., 2] = img[..., 0]
        return out

    def get_transform_init_args_names(self):
        return ()


class AverageFirstChannelToThird(ImageOnlyTransform):
    """Convert 2-channel image to 3-channel by averaging the first two channels."""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # img shape: (H, W, 2)
        if img.shape[-1] != 2:
            raise ValueError(f"Expected 2-channel image, got shape {img.shape}")
        h, w, _ = img.shape
        out = np.zeros((h, w, 3), dtype=img.dtype)
        out[..., 0] = img[..., 0]
        out[..., 1] = img[..., 1]
        out[..., 2] = (img[..., 0] + img[..., 1]) / 2
        return out

    def get_transform_init_args_names(self):
        return ()
