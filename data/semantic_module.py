from pathlib import Path
from functools import partial
from typing import Optional, Callable

import cv2
import albumentations as A
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from data.semantic_dataset import SemanticDataset
from data.image_tiling import slice_image_to_tiles


class SemanticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_path: str,
        classes: list[str],
        num_workers: int,
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
        transform: Optional[Callable] = None,
        return_filename: Optional[bool] = None,
        return_manufacturer: Optional[bool] = False,
        input_resolution: Optional[tuple[int, int]] = None,
        prob_channel_dropout: Optional[float] = 0.3,
        prob_channel_swap: Optional[float] = 0.3,
    ):
        """The Data Module specific to AOI data using 2 lighting conditions

        Args:
            batch_size (int): Batch sized used for training
            data_path (str): Path to the root of the dataset
            classes (list[str]): List of class names used
            num_workers (int): Number of DataLoader workers
            tile_size (Optional[int], optional): Tile size for cropping. Defaults to None.
            tile_overlap (Optional[int], optional): Overlap of the tiling. Defaults to None.
            transform (Optional[Callable], optional): Training augmentations to apply during. Defaults
                to None.
            return_filename (Optional[str], optional): Return additional filename. Defaults to None.
            input_resolution (Optional[tuple[int, int]], optional): Pad and/or crop to specifified
                input resolution. Defaults to None.
        """

        super().__init__()
        self.save_hyperparameters()

        self.stage = None
        self.classes = classes
        self.transform = transform
        self.tile_size = tile_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tile_overlap = tile_overlap
        self.return_filename = return_filename
        self.return_manufacturer = return_manufacturer
        # If networks can't handle variable input resolutions
        self.input_resolution = input_resolution

        self.prob_channel_dropout = prob_channel_dropout
        self.prob_channel_swap = prob_channel_swap

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

    def setup(self, stage: str):
        """Generate the type of the dataloaders depending on the stage.

        Args:
            stage (str): If 'tiling' the dataloaders return tiled versions specified
                by the user for predictions. 'eval' returns center crops with padding
                if needed for each split (including training). Otherwise, the training
                split is returned with passed `transform` augmentations and val and test
                with evaluation pad and center crop transforms.
        """
        self.stage = stage

        # Only center cropped if a input resolution is specified.
        if self.input_resolution:
            self.eval_transform = self.eval_transform

        if stage == "tiling":
            if not self.tile_overlap and not self.tile_size:
                raise TypeError(
                    "Tile overlap and size not properly configured for tiling."
                )

            self.eval_transform = partial(
                slice_image_to_tiles,
                tile_size=self.tile_size,
                overlap=self.tile_overlap,
            )

        self.aoi_train = SemanticDataset(
            Path(self.data_path) / "train",
            self.classes,
            transform=self.transform,
            prob_channel_dropout=self.prob_channel_dropout,
            prob_channel_swap=self.prob_channel_swap,
        )

        self.aoi_val = SemanticDataset(
            Path(self.data_path) / "val",
            self.classes,
            transform=self.eval_transform,
            tile_overlap=self.tile_overlap if stage == "tiling" else None,
            prob_channel_dropout=self.prob_channel_dropout,
            prob_channel_swap=self.prob_channel_swap,
        )

        self.aoi_test = SemanticDataset(
            Path(self.data_path) / "test",
            self.classes,
            transform=self.eval_transform,
            return_filename=self.return_filename,
            tile_overlap=self.tile_overlap if stage == "tiling" else None,
            prob_channel_dropout=self.prob_channel_dropout,
            prob_channel_swap=self.prob_channel_swap,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="spawn",
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
            batch_size=self.batch_size,
        )

    def teardown(self, stage: str):
        pass

    def custom_test_dataloader(self, path: str) -> DataLoader:
        """Create custom test dataset loader

        Args:
            path (str): Full path to the /test dataset with /img and /lbl dir.

        Returns:
            DataLoader: A dataloader with evaluation transformation set in the
                datamodule.
        """

        return DataLoader(
            SemanticDataset(
                path,
                self.classes,
                transform=self.eval_transform,
                return_filename=self.return_filename,
                tile_overlap=(self.tile_overlap if self.stage == "tiling" else None),
            ),
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
