from pathlib import Path
from typing import Optional, Callable

import torch
import numpy as np
from torch.utils.data import Dataset

from .common import list_buffer_image_pairs, load_buffer_pair_image


class PretrainDatasetiBot(Dataset):
    def __init__(
        self,
        path: str,
        return_filename: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.path = Path(path)
        self.prob_swap = 0.2
        self.transform = transform
        self.return_filename = return_filename

        image_root = self.path
        self.images = list_buffer_image_pairs(
            image_root,
            excluded_parts={"lbl", "label", "labels", "mask", "masks"},
        )

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        file, img0_path, img1_path = self.images[idx]
        img = load_buffer_pair_image(img0_path, img1_path)
        img0 = img[..., 0]

        assert not np.isnan(
            img
        ).any(), f"NaN values found in image {file} before mapping to [0, 1]"
        img = img / 255.0
        assert not np.isnan(
            img
        ).any(), f"NaN values found in image {file} after mapping to [0, 1]"

        # assert np.any(img >= 0) and np.any(img <= 1), f"Image {file} has values outside [0, 1] range"

        mask = (img0 > 0).astype(np.uint8)
        if self.transform is not None:
            # Expects Albumentations transforms
            transformed = self.transform(image=img, mask=mask, current_epoch=self.epoch)
            imgs = transformed["image"]
            masks = transformed["mask"]
        else:
            raise ValueError("Transform must be provided for the dataset")

        imgs = [torch.from_numpy(im).moveaxis(-1, -3).float() for im in imgs]
        masks = [torch.from_numpy(mask).bool() for mask in masks]

        for im in imgs:
            if im.isnan().any():
                raise ValueError(
                    f"NaN values found in image {file} after transformation"
                )

        return {"image": imgs, "file": file, "mask": masks}
