import os
import argparse
from pathlib import Path

# Base libraries
import cv2
import torch
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

# Dataset imports and utils
import segmentation.utils as utils
from pretrain.mae.lit_mae import LitMAE
from pretrain.dino.lit_mae_dino import LitMAEDino
from data.pretrain_module_dino import PretrainDataModuleDino
from data.retrieval_module import RetrievalDataModule
import warnings

# Evaluation imports
from retrieval.knn_seg import KNNSegmentation

# Imports to seed everything
import random
import numpy as np


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


class DataAugmentationDINO(object):

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        base_augmentations=False,
        normalize=False,
    ):

        normalization = A.Normalize(
            mean=[0.1570, 0.2096],  # Pretrain dataset mean
            std=[0.2221, 0.27701],  # Pretrain dataset std
        )

        self.normalization = normalization

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

    def __call__(self, image):

        crops = []
        crops.append(self.global_transfo1(image=image)["image"])
        crops.append(self.global_transfo2(image=image)["image"])
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image)["image"])
        return {"image": crops}


class KNNEvaluationCallback(pl.Callback):
    def __init__(
        self,
        batch_size,
        eval_data_path,
        classes,
        num_workers,
        input_resolution,
        eval_frequency,
    ):

        self.transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=input_resolution[0],
                    min_width=input_resolution[1],
                    # Avoids reflective padding
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                    p=1,
                ),
                A.CenterCrop(
                    input_resolution[0],
                    input_resolution[1],
                ),
            ]
        )
        self.eval_module = RetrievalDataModule(
            data_path=eval_data_path,
            batch_size=batch_size,
            classes=classes,
            num_workers=num_workers,
            train_size=0.7,
            return_manufacturer=True,
            return_device=True,
            input_resolution=(512, 512),
            normalize=config.normalize,
        )

        self.eval_module.setup(stage="fit")
        self.train_loader = self.eval_module.train_dataloader()
        self.val_dataloader = self.eval_module.val_dataloader()
        self.classes = classes
        self.eval_frequency = eval_frequency

    def on_train_epoch_end(self, trainer, pl_module: LitMAE):

        if trainer.current_epoch % self.eval_frequency != 0:
            # Skip evaluation if it's not the right epoch
            return

        # Perform evaluation here
        knn_model = KNNSegmentation(
            encoder=pl_module.get_encoder(),
            classes=self.classes,
            train_loader=self.train_loader,
            val_loader=self.val_dataloader,
            profile_memory=True,
        )

        mean_ious = knn_model.evaluate(
            k=3, distance_metric="cosine", weights="distance"
        )
        mean_iou = sum(mean_ious.values()) / len(mean_ious)
        pl_module.log("val/knn_mean_iou", mean_iou)


def run_segmentation(
    config: argparse.Namespace,
):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.seed)
    datamodule = PretrainDataModuleDino(
        batch_size=config.batch_size,
        test_batch_size=config.test_batch_size,
        data_path=config.data_path,
        num_workers=config.num_workers,
        input_resolution=(
            (config.model_params["img_size"], config.model_params["img_size"])
            if not None
            else None
        ),
        transform=DataAugmentationDINO(
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            local_crops_number=config.model_params["nlocal_crops"],
            base_augmentations=config.base_augmentations,
            normalize=config.normalize,
        ),
    )

    loggers = utils.get_loggers(
        job_id=config.job_id,
        tracking_uri=config.tracking_uri,
        logging_path=str(config.path.absolute()),
        run_name=config.run_name,
        experiment_n=config.experiment_n,
        experiment_name=config.name,
        use_tb_logger=False,
    )

    model = LitMAEDino(parameters=config.model_params)

    if config.monitor == "val/knn_mean_iou":
        print("Using KNN mean IoU as monitor metric.")

    # Create callbacks
    ckpt_callback = ModelCheckpoint(
        save_top_k=5,
        mode="max" if config.monitor == "val/knn_mean_iou" else "min",
        monitor=config.monitor,
        dirpath=str(config.path),
        auto_insert_metric_name=False,
        filename="epoch={epoch}-val_knn_mean_iou={val/knn_mean_iou:.3f}",
    )

    callbacks = [
        ckpt_callback,
        EarlyStopping(
            patience=config.patience,
            monitor=config.monitor,
            mode="max" if config.monitor == "val/knn_mean_iou" else "min",
            verbose=True,
        ),
        utils.GradNormCallback(),
        KNNEvaluationCallback(
            batch_size=config.test_batch_size,
            eval_data_path=config.eval_data_path,
            classes=config.classes,
            num_workers=config.num_workers,
            input_resolution=config.input_resolution,
            eval_frequency=config.knn_eval_frequency,
        ),
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
        ),
    ]

    if hasattr(config, "accumulate_grad_batches"):
        warnings.warn(
            f"Accumulating gradients over {config.accumulate_grad_batches} batches."
        )

    check_val_every_n_epoch = (
        config.check_val_every_n_epoch
        if hasattr(config, "check_val_every_n_epoch")
        else 1
    )
    if "knn_eval_frequency" in vars(config):
        check_val_every_n_epoch = config.knn_eval_frequency

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        strategy=(
            config.strategy
            if config.strategy == "auto"
            else DDPStrategy(find_unused_parameters=True)
        ),
        default_root_dir=str(config.path),
        log_every_n_steps=config.log_every_n,
        gradient_clip_val=None,  # Disable automatic gradient clipping as we're using manual optimization
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_checkpointing=True,
        num_sanity_val_steps=5,
        precision=config.precision,
        sync_batchnorm=True,
        accumulate_grad_batches=(
            config.accumulate_grad_batches
            if hasattr(config, "accumulate_grad_batches")
            else None
        ),
        devices="auto",
        accelerator="gpu",
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.ckpt_path if hasattr(config, "ckpt_path") else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_key",
        type=str,
        required=True,
        help="MLFlow run name suffix",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Experiment seed",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./scripts/logs/",
        help="Path to log the training files, e.g. checkpoints",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/pretrain/mae.yml",
        help="Path to model specific YAML file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/pretrain/main.yml",
        help="Path to experiment specific YAML file",
    )

    parser.add_argument(
        "--data_path",
        type=str,
    )

    parser.add_argument(
        "--eval_data_path",
        type=str,
    )

    parser.add_argument(
        "--data_collection_path",
        type=str,
        help="Path to log the training files, e.g. checkpoints",
        default=None,
    )

    # Training parameters
    # Overides the config if passed.
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--epochs",
        type=int,
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="Batch size for testing",
    )

    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to model checkpoint"
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before backprop. This is to support smaller patch sizes.",
    )

    parser.add_argument(
        "--vit_type",
        type=str,
        default=None,
        help="Type of ViT model to use",
    )

    parser.add_argument(
        "--reconstruction_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the reconstruction loss",
    )

    parser.add_argument(
        "--dino_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the dino loss",
    )

    parser.add_argument(
        "--student_temperature",
        type=float,
        default=0.1,
        help="Temperature for the student model",
    )

    parser.add_argument(
        "--teacher_temperature",
        type=float,
        default=0.04,
        help="Temperature for the teacher model",
    )

    parser.add_argument(
        "--teacher_momentum",
        type=float,
        default=0.996,
        help="Momentum for the teacher model",
    )

    parser.add_argument(
        "--base_augmentations",
        action="store_true",
        help="Use old augmentations for DINO pretraining",
        default=None,
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply normalization to the images",
        default=None,
    )

    parser.add_argument("--knn_eval_frequency", type=int, default=None)

    parser.add_argument(
        "--monitor",
        type=str,
    )

    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    # Initialize save directory
    args.path = Path(args.path)
    args.job_id = os.environ.get("SLURM_JOB_ID", "")

    # Create repository after merging name from args and config
    config = utils.merge_config(args.config, args, args.model_config)
    config.path = args.path / f"{config.name}"
    config.path.mkdir(exist_ok=True, parents=True)
    config.run_name = f"{config.job_id}-{config.run_key}"

    run_segmentation(config)
