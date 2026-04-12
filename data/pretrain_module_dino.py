import os
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import albumentations as A
import lightning.pytorch as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from data import PretrainDataset





class PretrainDataModuleDino(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        test_batch_size: int,
        data_path: str,
        num_workers: int,
        transform: Optional[Callable] = None,
        return_filename: Optional[bool] = None,
        input_resolution: Optional[tuple[int, int]] = None,
        seed: Optional[int] = 42,
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

        #Note: stage not implemented yet
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

        self.aoi_train = PretrainDataset(
            Path(self.data_path) / "train",
            transform=self.transform,
            return_filename=self.return_filename,
        )

        self.aoi_val = PretrainDataset(
            Path(self.data_path) / "val",
            transform=self.transform,
            return_filename=self.return_filename,
        )
        
        self.aoi_test = PretrainDataset(
            Path(self.data_path) / "test",
            transform=self.eval_transform,
            return_filename=self.return_filename
        )

    def train_dataloader(self) -> DataLoader:

        #sampler = DistributedSampler(self.aoi_train, shuffle=True, seed=self.seed) if self.trainer and self.trainer.strategy in ["ddp", "ddp_spawn"] else None
        #print(f"Sampler: {sampler}")

        return DataLoader(
            self.aoi_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.aoi_val,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_test,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
        )

    def teardown(self, stage: str):
        pass
