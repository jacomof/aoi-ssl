import os
import argparse
from pathlib import Path

import cv2
import torch
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from pretrain.mae.mae_fastervit import MaskedAutoencoderFasterVit
from pretrain.mae.mae import MaskedAutoencoder
import segmentation.utils as utils
from segmentation.models.faster_vit.lit_faster_vit_seg import LitFasterViTSegmentation
from segmentation.models.vit.lit_vit_seg import LitViTSegmentation
from pretrain.mae.lit_mae import LitMAE
from pretrain.dino.lit_mae_dino import LitMAEDino
from data.semantic_module import SemanticDataModule
from pretrain.dino.pretrain_dino import DataAugmentationDINO, GaussianBlur, Solarization, ColorJitterFor2Channel, ClipTo01
from data.pretrain_module_ibot import DataAugmentationiBot
from data.retrieval_module import RetrievalDataModule
from segmentation.models.vit import init_vit

def run_segmentation(
    config: argparse.Namespace,
    seg_model: LitFasterViTSegmentation | LitViTSegmentation,
):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.seed)

    # Fixed augmentation pipeline by Joris/Thomas
    crop_size_0 = tuple(config.input_resolution)[0]
    crop_size_1 = tuple(config.input_resolution)[1]
    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=crop_size_0,
                min_width=crop_size_1,
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
            A.OneOrOther(
                A.CropNonEmptyMaskIfExists(crop_size_0, crop_size_1),
                A.RandomCrop(crop_size_0, crop_size_1),
                p=0.8,
            ),
        ]
    )

    # datamodule = RetrievalDataModule(
    #     config.data_path,
    #     config.batch_size,
    #     config.classes,
    #     transform=transform,
    #     num_workers=config.num_workers,
    #     input_resolution=tuple(config.input_resolution) if not None else None,
    #     random_seed=config.seed,
    #     normalize=False
    # )

    datamodule = SemanticDataModule(
        config.batch_size,
        config.data_path,
        config.classes,
        transform=transform,
        num_workers=config.num_workers,
        input_resolution=tuple(config.input_resolution) if not None else None,
        prob_channel_dropout=0.0,
        prob_channel_swap=0.0,
    )

    datamodule.setup(stage="fit")
    print("Data module setup complete.")

    loggers = utils.get_loggers(
        job_id=config.job_id,
        tracking_uri=config.tracking_uri,
        logging_path=str(config.path.absolute()),
        run_name=config.run_name,
        experiment_n=config.experiment_n,
        experiment_name=config.name,
    )

    

    if hasattr(config, "use_dinov2") and config.use_dinov2:
        # For training DinoV2
        print("Using DinoV2 encoder")
        encoder = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_vits14'
        )
        print("DinoV2 info: \n", encoder)
        model = seg_model(parameters=config.model_params, classes=config.classes)

        model.model.set_encoder(
            encoder, freeze_encoder=False, use_dinov2=True
        )
    elif (
        hasattr(config, "encoder_ckpt_path")
        and hasattr(config, "encoder_ckpt_model_params")
        and config.encoder_ckpt_path
        and config.encoder_ckpt_model_params
    ):
        print("Using pretrained encoder from checkpoint.")
        # For pretrained ViT
        config_mae = utils.merge_config(
            config.config, argparse.Namespace(), config.encoder_ckpt_model_params
        )
        # Merge so that segmentation model can use the same parameters
        # as the encoder
        config.model_params = utils.merge_model_params_segementation_encoder(
            config, config.encoder_ckpt_model_params
        )

        encoder_cls_map = {
            "mae": MaskedAutoencoder,
            "fastervit": MaskedAutoencoderFasterVit,
        }
        encoder_cls = encoder_cls_map.get(config.model, MaskedAutoencoder)
        print("Configuration after merge: ", config)
        mae = LitMAE.load_from_checkpoint(
            config.encoder_ckpt_path, parameters=config_mae.model_params,
            model=encoder_cls,
            strict=False
        )
        encoder = mae.get_encoder()
        config.model_params["embed_dim"] = encoder.embed_dim
        model = seg_model(parameters=config.model_params, classes=config.classes)

        # Moved set_encoder outside of constructor so that the encoder is not initialized two times
        model.model.set_encoder(encoder=encoder, freeze_encoder=False)

        print("Successfully loaded MAE model.")

    else:
        print("Using ViT from scratch.")
        # For training from scratch
        model = seg_model(parameters=config.model_params, classes=config.classes)
        encoder = init_vit(
            config.model_params["vit_type"],
            num_register_tokens=config.model_params["reg_tokens"],
            **config.model_params
        )
        model.model.set_encoder(encoder=encoder, freeze_encoder=False)
        print("Successfully initialized segmentation model.")

    # Create callbacks
    ckpt_callback = ModelCheckpoint(
        save_top_k=5,
        mode="min",
        every_n_train_steps=10,
        monitor=config.monitor,
        dirpath=str(config.path),
        auto_insert_metric_name=False,
        filename="epoch={epoch}-val_loss={train/loss:.2f}-val_iou={val/iou:.2f}",
    )
    early_stop_callback = EarlyStopping(
        patience=config.patience, monitor=config.monitor, mode="min"
    )

    learning_rate_callback = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    callbacks = [
        ckpt_callback,
        early_stop_callback,
        utils.GradNormCallback(),
        learning_rate_callback,
    ]

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
        check_val_every_n_epoch=config.check_val_every_n,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_checkpointing=True,
        num_sanity_val_steps=5,
        precision=config.precision,
        sync_batchnorm=True,
        accumulate_grad_batches=config.accumulate_grad_batches
        if hasattr(config, "accumulate_grad_batches")
        else 1,
    )

    print("About to launch training loop!")

    trainer.fit(model=model, datamodule=datamodule)

    # Do final test
    trainer.test(ckpt_path="best", datamodule=datamodule)


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
        default="./configs/experiment/1/faster_vit_4.yml",
        help="Path to model specific YAML file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/experiment/1/main.yml",
        help="Path to experiment specific YAML file",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        type=str,
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
    args = parser.parse_args()

    # Initialize save directory
    args.path = Path(args.path)
    args.job_id = os.environ.get("SLURM_JOB_ID", "")

    # Create repository after merging name from args and config
    config = utils.merge_config(args.config, args, args.model_config)

    # model saved in args.path + experiment_name + job_id
    config.path = args.path / f"{config.name}"
    config.path.mkdir(exist_ok=True, parents=True)

    # Select model
    models = {
        "fastervit": LitFasterViTSegmentation,
        "vit": LitViTSegmentation,
    }
    print("Config.model is: ", config.model)
    print("Final configuration: ", models[config.model.lower()])

    config.run_name = f"{args.model.lower()}-{config.run_key}"
    run_segmentation(config, models[config.model.lower()])
