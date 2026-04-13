import os
from pathlib import Path
from typing import Optional, Callable
import warnings

import cv2
import numpy as np
import albumentations as A
import lightning.pytorch as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.pretrain_dataset import PretrainDataset


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        test_batch_size: int,
        data_path: str,
        num_workers: int,
        transform: Optional[Callable] = None,
        return_filename: Optional[bool] = None,
        input_resolution: Optional[tuple[int, int]] = None,
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
        self.transform = transform
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.return_filename = return_filename

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

        data_root = Path(self.data_path)
        train_root = data_root / "train"
        val_root = data_root / "val"
        test_root = data_root / "test"

        if not val_root.exists() and (data_root / "eval").exists():
            val_root = data_root / "eval"
            warnings.warn(
                f"Validation split not found at {data_root / 'val'}; using {val_root} instead."
            )

        if not test_root.exists() and val_root.exists():
            test_root = val_root
            warnings.warn(
                f"Test split not found at {data_root / 'test'}; using {test_root} instead."
            )

        self.aoi_train = PretrainDataset(
            train_root,
            transform=self.transform,
            return_filename=self.return_filename,
        )

        self.aoi_val = PretrainDataset(
            val_root,
            transform=self.eval_transform,
            return_filename=self.return_filename,
        )

        self.aoi_test = PretrainDataset(
            test_root,
            transform=self.eval_transform,
            return_filename=self.return_filename,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_test,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
        )

    def teardown(self, stage: str):
        pass
