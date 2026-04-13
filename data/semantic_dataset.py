from __future__ import annotations

from pathlib import Path
import random

from torch.utils.data import Dataset
import numpy as np
import torch

from .common import (
    list_buffer_image_pairs,
    load_buffer_pair_image,
    find_label_for_image,
    load_mask,
    apply_transform,
    to_image_tensor,
    to_class_mask_tensor,
)


class SemanticDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        classes: list[str],
        transform=None,
        return_filename: bool | None = False,
        tile_overlap=None,
        prob_channel_dropout: float = 0.0,
        prob_channel_swap: float = 0.0,
    ):
        self.root = Path(root)
        self.classes = classes
        self.transform = transform
        self.return_filename = bool(return_filename)
        self.tile_overlap = tile_overlap
        self.prob_channel_dropout = float(prob_channel_dropout)
        self.prob_channel_swap = float(prob_channel_swap)

        img_dir = self.root / "img"
        image_root = img_dir if img_dir.exists() else self.root
        self.images = list_buffer_image_pairs(
            image_root,
            excluded_parts={"lbl", "label", "labels", "mask", "masks"},
        )

    def __len__(self) -> int:
        return len(self.images)

    def _apply_channel_regularization(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] < 2:
            return image

        out = image.copy()
        if random.random() < self.prob_channel_swap:
            out[..., [0, 1]] = out[..., [1, 0]]

        if random.random() < self.prob_channel_dropout:
            drop_idx = random.choice([0, 1])
            out[..., drop_idx] = 0

        return out

    def __getitem__(self, idx: int) -> dict:
        sample_stem, channel_0_path, channel_1_path = self.images[idx]
        image = load_buffer_pair_image(channel_0_path, channel_1_path)
        image = self._apply_channel_regularization(image)

        h, w = image.shape[:2]
        base_image_path = channel_0_path.with_name(
            f"{sample_stem}{channel_0_path.suffix}"
        )
        mask_path = find_label_for_image(base_image_path) or find_label_for_image(
            channel_0_path
        )
        class_mask = load_mask(mask_path, h, w, num_classes=len(self.classes))

        image, class_mask = apply_transform(self.transform, image, class_mask)

        class_mask_tensor = to_class_mask_tensor(class_mask)
        sample = {
            "image": to_image_tensor(image),
            "class_mask": class_mask_tensor,
            "ignore_mask": torch.zeros_like(class_mask_tensor),
        }

        if self.return_filename:
            sample["name"] = channel_0_path.name

        return sample
