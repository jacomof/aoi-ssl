import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import albumentations as A
import lightning.pytorch as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.pretrain_dataset_ibot import PretrainDatasetiBot

import random
import math


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Translations, rotations, scaling - Test time adaptations
class GaussianBlur(A.GaussianBlur):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, radius_min=0.1, radius_max=2.0, **kwargs):
        super().__init__(sigma_limit=(radius_min, radius_max), **kwargs)


class Solarization(A.ImageOnlyTransform):
    """
    Apply Solarization to the PIL image.

    Remove
    """

    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = p

    def apply(self, img, **params):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        else:
            return A.solarize(img)


class ColorJitterFor2Channel(A.ImageOnlyTransform):
    """
    Apply ColorJitter to 2-channel images by adding a dummy channel.
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8):
        super().__init__(always_apply=False, p=p)
        self.color_jitter = A.Compose(
            [
                A.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ]
        )

    def apply(self, img, **params):
        # Add a dummy channel to make it 3-channel
        if img.shape[-1] == 2:  # Check if the image has 2 channels
            dummy_channel = np.zeros_like(img[..., :1])  # Create a dummy channel
            img = np.concatenate([img, dummy_channel], axis=-1)  # Add the dummy channel

        # Apply ColorJitter
        img = self.color_jitter(image=img)["image"]

        # Remove the dummy channel
        img = img[..., :2]  # Keep only the first two channels
        return img


class ClipTo01(A.ImageOnlyTransform):
    def apply(self, img, **params):
        return img.clip(0.0, 1.0).astype(np.float32)


class DataAugmentationiBot(object):

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        patch_size,
        use_swin=False,
        mim_start_epoch=0,
        base_augmentations=False,
        normalize=False,
    ):

        normalization = A.Normalize(
            mean=[0.1570, 0.2096],  # Pretrain dataset mean
            std=[0.2221, 0.27701],  # Pretrain dataset std
        )

        self.current_epoch = 0
        self.mim_start_epoch = mim_start_epoch

        self.normalization = normalization

        self.use_swin = use_swin

        self.patch_size = patch_size
        if use_swin:
            self.patch_size = patch_size * 8  # 3 downsamples of stride 2 -> (2**3)

        flip_and_color_jitter = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                ColorJitterFor2Channel(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
            ]
        )

        old_augmentations1 = [
            flip_and_color_jitter,
            A.GaussianBlur(p=1.0),
        ]

        old_augmentations2 = [
            flip_and_color_jitter,
            A.GaussianBlur(p=0.1),
            Solarization(p=0.2),
        ]

        new_augmentations1 = [
            A.SomeOf(
                [
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.GaussianBlur(sigma_limit=0.75),
                    A.RandomContrast(),
                    A.GaussNoise(var_limit=(0.05, 0.05 * 255)),
                    A.RandomRotate90(),
                    A.Affine(scale=(0.5, 2.0)),
                    A.Affine(rotate=(-45, 45)),
                    A.Affine(shear=(-8, 8)),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                ],
                3,
            ),
        ]

        new_augmentations2 = [
            A.SomeOf(
                [
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.GaussianBlur(sigma_limit=0.75),
                    A.GaussNoise(var_limit=(0.05, 0.05 * 255)),
                    A.RandomRotate90(),
                    A.Affine(scale=(0.5, 2.0)),
                    A.Affine(rotate=(-45, 45)),
                    A.Affine(shear=(-8, 8)),
                ],
                3,
            ),
        ]

        augmentations1 = (
            old_augmentations1 if base_augmentations else new_augmentations1
        )
        augmentations2 = (
            old_augmentations2 if base_augmentations else new_augmentations2
        )

        if normalize:
            augmentations1.append(normalization)
            augmentations2.append(normalization)

        augmentations1 = A.Compose(augmentations1)
        augmentations2 = A.Compose(augmentations2)

        # first global crop
        self.global_transfo1 = A.Compose(
            [
                A.RandomResizedCrop(
                    224, 224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC
                ),
                ClipTo01(p=1.0),  # Clip to [0, 1] range
                augmentations1,
            ]
        )
        # second global crop
        self.global_transfo2 = A.Compose(
            [
                A.RandomResizedCrop(
                    224, 224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC
                ),
                ClipTo01(p=1.0),  # Clip to [0, 1] range
                augmentations2,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = A.Compose(
            [
                A.RandomResizedCrop(
                    96, 96, scale=local_crops_scale, interpolation=cv2.INTER_CUBIC
                ),
                ClipTo01(p=1.0),  # Clip to [0, 1] range
                augmentations1,
            ]
        )

    def input_mask_generator(self, img, patch_size=16, prediction_ratio=0.5):
        """Generate input masks for the iBot model.
        This method generates masks for the input images based on the iBot masking strategy.
        It creates a mask for each image in the output, where the mask is a boolean array
        indicating which patches are masked (True) and which are not (False).
        The masking strategy is designed to randomly mask a certain percentage of patches
        in the image, following the iBot approach.
        Args:
            output (list): A list of images, where each image is a tensor of shape (batch_size, channels, height, width).
        Returns:
            list: A list of boolean masks, where each mask has the same height and width as the corresponding image,
                  and indicates which patches are masked.
        """

        log_aspect_ratio = (math.log(0.3), math.log(1 / 0.3))  # Example values
        H, W = img.shape[0] // patch_size, img.shape[1] // patch_size

        high = prediction_ratio * H * W

        # following BEiT (https://arxiv.org/abs/2106.08254), see at
        # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
        mask = np.zeros((H, W), dtype=bool)
        mask_count = 0
        while mask_count < high:
            max_mask_patches = high - mask_count

            delta = 0
            for attempt in range(10):
                low = (min(H, W) // 3) ** 2
                target_area = random.uniform(low, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < W and h < H:
                    top = random.randint(0, H - h)
                    left = random.randint(0, W - w)

                    num_masked = mask[top : top + h, left : left + w].sum()
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                if delta > 0:
                    break

            if delta == 0:
                break
            else:
                mask_count += delta

        return mask.flatten()

    def __call__(self, image, mask, current_epoch):
        crops = []
        crops.append(self.global_transfo1(image=image, mask=mask)["image"])
        crops.append(self.global_transfo2(image=image, mask=mask)["image"])
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image, mask=mask)["image"])

        masks = []
        for i in range(2):
            should_mask = random.random() < 0.5
            if (
                should_mask and current_epoch >= self.mim_start_epoch
            ):  # Start masking after epoch 50
                prediction_ratio = random.uniform(0.1, 0.5)
                masks.append(
                    self.input_mask_generator(
                        crops[i],
                        patch_size=self.patch_size,
                        prediction_ratio=prediction_ratio,
                    )
                )
            else:
                prediction_ratio = 0.0
                masks.append(
                    self.input_mask_generator(
                        crops[i],
                        patch_size=self.patch_size,
                        prediction_ratio=prediction_ratio,
                    )
                )

        for i in range(2, len(crops)):
            masks.append(
                self.input_mask_generator(
                    crops[i], patch_size=self.patch_size, prediction_ratio=0
                )
            )
        return {"image": crops, "mask": masks}


class PretrainDataModuleiBot(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        test_batch_size: int,
        data_path: str,
        num_workers: int,
        return_filename: Optional[bool] = None,
        input_resolution: Optional[tuple[int, int]] = None,
        seed: Optional[int] = 42,
        global_crops_scale: Optional[tuple[float, float]] = (0.4, 1.0),
        local_crops_scale: Optional[tuple[float, float]] = (0.05, 0.4),
        local_crops_number: Optional[int] = 6,
        patch_size: Optional[int] = 16,
        use_swin: Optional[bool] = False,
        mim_start_epoch: Optional[int] = 0,
        base_augmentations: Optional[bool] = False,
        normalize: Optional[bool] = False,
    ):
        """The Data Module specific to AOI data using 2 lighting conditions

        Args:
            batch_size (int): Batch sized used for training
            data_path (str): Path to the root of the dataset
            num_workers (int): Number of DataLoader workers
            transform (Optional[Callable], optional): Training augmentations to apply during. Defaults
                to None.
            return_filename (Optional[str], optional): Return additional filename. Defaults to None.
            input_resolution (Optional[tuple[int, int]], optional): Pad and/or crop to specifified
                input resolution. Defaults to None.
        """

        super().__init__()
        self.save_hyperparameters()

        self.stage = None
        self.transform = DataAugmentationiBot(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            patch_size=patch_size,
            use_swin=use_swin,
            mim_start_epoch=mim_start_epoch,
            base_augmentations=base_augmentations,
            normalize=normalize,
        )
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.return_filename = return_filename
        self.seed = seed

        # TODO: Configurable lighting conditions

        # If networks can't handle variable input resolutions
        self.input_resolution = input_resolution

        if self.input_resolution:
            self.eval_transform = A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=self.input_resolution[0],
                        min_width=self.input_resolution[1],
                        # Avoids reflective padding
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                        p=1,
                    ),
                    A.CenterCrop(
                        self.input_resolution[0],
                        self.input_resolution[1],
                    ),
                ]
            )
        else:
            # No cropping during evaluation for convolutional models
            self.eval_transform = None

        self.current_epoch = 0

    # TODO: Implement stage specific dataloaders, if needed
    def setup(self, stage: str = None):
        """Generate the type of the dataloaders depending on the stage.

        Args:
            stage (str): If 'tiling' the dataloaders return tiled versions specified
                by the user for predictions. 'eval' returns center crops with padding
                if needed for each split (including training). Otherwise, the training
                split is returned with passed `transform` augmentations and val and test
                with evaluation pad and center crop transforms.
        """

        # Note: stage not implemented yet
        self.stage = stage

        # Only center cropped if a input resolution is specified.
        if self.input_resolution:
            self.eval_transform = self.eval_transform

        # Option to uniformly sample over the different customer subsets
        self.train_sampler = None
        if "sample_weights.npy" in os.listdir(self.data_path):
            sample_weights = np.load(Path(self.data_path) / "sample_weights.npy")
            self.train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

        self.aoi_train = PretrainDatasetiBot(
            Path(self.data_path) / "train",
            transform=self.transform,
            return_filename=self.return_filename,
        )

        self.aoi_val = PretrainDatasetiBot(
            Path(self.data_path) / "val",
            transform=self.transform,
            return_filename=self.return_filename,
        )

        self.aoi_test = PretrainDatasetiBot(
            Path(self.data_path) / "test",
            transform=self.eval_transform,
            return_filename=self.return_filename,
        )

    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.aoi_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.aoi_val, num_workers=self.num_workers, batch_size=self.test_batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_test,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
        )

    def teardown(self, stage: str):
        pass

    def set_epoch(self, epoch: int):
        """Set the current epoch for the data augmentation transformations."""
        self.current_epoch = epoch
        self.aoi_train.set_epoch(epoch)
        self.aoi_val.set_epoch(epoch)
        self.aoi_test.set_epoch(epoch)
