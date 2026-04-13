import os
import argparse
from pathlib import Path

import cv2
import torch
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import segmentation.utils as utils
from pretrain.mae.lit_mae import LitMAE
from pretrain.mae.mae_fastervit import MaskedAutoencoderFasterVit
from data.pretrain_module import PretrainDataModule
import warnings

from retrieval.knn_seg import KNNSegmentation
from data.retrieval_module import RetrievalDataModule


class KNNEvaluationCallback(pl.Callback):
    def __init__(
        self,
        batch_size,
        eval_data_path,
        classes,
        num_workers,
        input_resolution,
        eval_frequency,
        random_seed=42,
    ):

        self.random_seed = random_seed

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
            classes=["wire", "ball", "wedge", "epoxy"],
            num_workers=num_workers,
            train_size=0.7,
            return_manufacturer=True,
            return_device=True,
            input_resolution=(512, 512),
            normalize=config.normalize,
            random_seed=random_seed,
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

    transforms = [
        A.PadIfNeeded(
            min_height=config.crop_size,
            min_width=config.crop_size,
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
        A.RandomCrop(config.crop_size, config.crop_size),
    ]
    normalize = config.normalize if hasattr(config, "normalize") else False
    if normalize:
        transforms.append(
            A.Normalize(
                mean=[0.1570, 0.2220],  # Pretrain dataset mean
                std=[0.2096, 0.27701],  # Pretrain dataset std
            )
        )

    datamodule = PretrainDataModule(
        config.batch_size,
        config.test_batch_size,
        config.data_path,
        num_workers=config.num_workers,
        input_resolution=(
            (config.model_params["img_size"], config.model_params["img_size"])
            if not None
            else None
        ),
        transform=A.Compose(transforms),
    )

    # In pretrain.py, replace the logger creation with exact copy from pretrain_dino.py
    loggers = utils.get_loggers(
        job_id=config.job_id,
        tracking_uri=config.tracking_uri,
        logging_path=str(config.path.absolute()),
        run_name=config.run_name,
        experiment_n=config.experiment_n,
        experiment_name=config.name,
    )

    if config.model_params.get("use_fastervit_0", False):
        print("Using FasterViT 0.0 as the encoder.")
        model = LitMAE(parameters=config.model_params, model=MaskedAutoencoderFasterVit)
    else:
        print("Using standard ViT as the encoder.")
        model = LitMAE(parameters=config.model_params)

    # Create callbacks
    ckpt_callback = ModelCheckpoint(
        save_top_k=5,
        # if using MAE loss, monitor val/loss, otherwise monitor val/knn_mean_iou
        mode="max" if config.monitor == "val/knn_mean_iou" else "min",
        monitor=config.monitor,
        dirpath=str(config.path),
        auto_insert_metric_name=False,
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
    )

    callbacks = [
        ckpt_callback,
        EarlyStopping(
            patience=config.patience,
            monitor=config.monitor,
            mode="max" if config.monitor == "val/knn_mean_iou" else "min",
            verbose=True,
            check_on_train_epoch_end=False,
        ),
        utils.GradNormCallback(),
    ]

    if hasattr(config, "accumulate_grad_batches"):
        warnings.warn(
            f"Accumulating gradients over {config.accumulate_grad_batches} batches."
        )

    check_val_every_n_epoch = getattr(
        config,
        "check_val_every_n",
        getattr(config, "check_val_every_n_epoch", 1),
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
        gradient_clip_val=config.gradient_clip_val,
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
            else 1
        ),
        accelerator="gpu",
    )
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=getattr(config, "ckpt_path", None)
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
        help="Number of batches to accumulate before backprop. This is to support smaller patch sizes.",
    )

    parser.add_argument(
        "--vit_type",
        type=str,
        default=None,
        help="Type of ViT model to use",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=None,
        help="Whether to normalize the input images.",
    )

    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=None,
        help="Mask ratio for the MAE model. If not set, defaults to 0.75.",
    )

    parser.add_argument("--monitor", type=str, default=None)

    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    # Initialize save directory
    args.path = Path(args.path)
    args.job_id = os.environ.get("SLURM_JOB_ID", "")

    # Create repository after merging name from args and config
    config = utils.merge_config(args.config, args, args.model_config)
    config.path = args.path / f"{config.name}"
    path = Path(config.path)
    config.path.mkdir(exist_ok=True, parents=True)
    config.run_name = f"{config.job_id}-{config.run_key}"

    # Add model parameters to config
    if hasattr(config, "mask_ratio"):
        config.model_params["mask_ratio"] = (
            config.mask_ratio if config.mask_ratio else 0.75
        )
    if hasattr(config, "lr"):
        config.model_params["lr"] = config.lr if config.lr else 1e-4

    run_segmentation(config)
