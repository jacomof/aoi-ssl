from pathlib import Path
from functools import partial
from typing import Optional, Callable
import numpy as np

import cv2
import albumentations as A
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from data.retrieval_dataset import RetrievalDataset
from data.image_tiling import slice_image_to_tiles


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_split_images(split_path: Path) -> list[Path]:
    if not split_path.exists():
        return []
    images = []
    for p in split_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem_lower = p.stem.lower()
        if stem_lower.endswith("_mask") or stem_lower.endswith("_label"):
            continue
        images.append(p)
    return sorted(images)


def _list_images_from_split_root(split_root: Path) -> list[Path]:
    """List split images from explicit split layout: <split>/img/*."""
    return _list_split_images(split_root / "img")


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        classes: list[str],
        num_workers: int,
        train_size: float = 0.85,
        img_dir: str = "train",
        holdout_dir: str = "test",
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
        transform: Optional[Callable] = None,
        return_filename: Optional[bool] = None,
        return_manufacturer: Optional[bool] = False,
        return_device: Optional[bool] = False,
        input_resolution: Optional[tuple[int, int]] = None,
        augmented: bool = False,
        normalize: bool = False,
        random_seed: int = 42,
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

        print("Initializing RetrievalDataModule...")
        super().__init__()
        self.save_hyperparameters()

        self.stage = None
        self.classes = classes
        self.transform = transform
        self.tile_size = tile_size
        self.data_path = Path(data_path)
        self.img_dir = img_dir
        self.holdout_dir = holdout_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tile_overlap = tile_overlap
        self.return_filename = return_filename
        self.return_manufacturer = return_manufacturer
        self.return_device = return_device
        # If networks can't handle variable input resolutions
        self.input_resolution = input_resolution

        self.train_size = train_size
        self.augmented = augmented
        self.normalize = normalize
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        print("Augmented is true!")

        # Channel 0 mean: 0.17811504349642812, std: 0.22236312176409964
        # Channel 1 mean: 0.23459850405723615, std: 0.270697137739698
        self.normalization = A.Normalize(
            mean=[0.17812, 0.23460],
            std=[0.22236, 0.27070],
        )


        train_split_img = self.data_path / "train" / "img"
        val_split_img = self.data_path / "val" / "img"
        test_split_img = self.data_path / "test" / "img"

        use_explicit_splits = (
            train_split_img.exists() and val_split_img.exists() and test_split_img.exists()
        )

        if use_explicit_splits:
            # Preferred dataset layout:
            # data_path/train/{img,lbl}, data_path/val/{img,lbl}, data_path/test/{img,lbl}
            self.train_img_list = _list_images_from_split_root(self.data_path / "train")
            self.val_img_list = _list_images_from_split_root(self.data_path / "val")
            self.holdout_img_list = _list_images_from_split_root(self.data_path / "test")
            print(
                "Using explicit splits from train/val/test directories: "
                f"train={len(self.train_img_list)}, "
                f"val={len(self.val_img_list)}, "
                f"test={len(self.holdout_img_list)}"
            )
        else:
            # Legacy fallback: train/val random split from a single image directory
            # and holdout from a separate directory.
            img_list = _list_split_images(self.data_path / self.img_dir)
            self.holdout_img_list = _list_split_images(self.data_path / self.holdout_dir)

            print(self.train_size)
            print(len(img_list))
            np.random.shuffle(img_list)
            self.train_img_list = img_list[: int(len(img_list) * self.train_size)]
            self.val_img_list = img_list[int(len(img_list) * self.train_size):]

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

        if transform:
            # If the transform is provided, it will be used for training
            print("Using provided transform for training.")
            self.transform = transform

        elif augmented:
            self.transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.input_resolution[0],
                    min_width=self.input_resolution[1],
                    # Avoids reflective padding
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                    p=1,
                ),
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
                A.RandomCrop(self.input_resolution[0], self.input_resolution[1]),
            ]
        )
        else:
            self.transform = A.Compose(
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
        
        if self.normalize:
            self.transform = A.Compose(
                [
                    self.transform,
                    self.normalization,
                ]
            )

            self.eval_transform = A.Compose(
                [
                    self.eval_transform,
                    self.normalization,
                ]
            )



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

        self.aoi_train = RetrievalDataset(
            self.train_img_list,
            self.classes,
            transform=self.transform,
            return_filename=self.return_filename,
            return_manufacturer=self.return_manufacturer,
            return_device=self.return_device,
        )

        self.aoi_val = RetrievalDataset(
            self.val_img_list,
            self.classes,
            transform=self.eval_transform,
            return_filename=self.return_filename,
            return_device=self.return_device,
            return_manufacturer=self.return_manufacturer,
        )

        self.aoi_test = RetrievalDataset(
            self.holdout_img_list,
            self.classes,
            transform=self.eval_transform,
            return_filename=self.return_filename,
            return_device=self.return_device,
            return_manufacturer=self.return_manufacturer,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.aoi_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=False,
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
