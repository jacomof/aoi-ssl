
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
from pretrain.mae.mae_fastervit import MaskedAutoencoderFasterVit

# Dataset imports and utils
import segmentation.utils as utils
from pretrain.mae.lit_mae import LitMAE
from data.pretrain_module import PretrainDataModule
from data.retrieval_module import RetrievalDataModule  
import warnings

# Evaluation imports
from retrieval.knn_seg import KNNSegmentation
from retrieval.knn_seg_hbird import KNNHummingBirdSegmentation


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class KNNEvaluationCallback(pl.Callback):
    def __init__(self, batch_size, eval_data_path, classes, num_workers, input_resolution, eval_frequency):

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
            normalize=config.normalize
        )
        
        self.eval_module.setup(stage="fit")
        self.train_loader = self.eval_module.train_dataloader()
        self.val_dataloader = self.eval_module.val_dataloader()
        self.classes = classes
        self.eval_frequency = eval_frequency
        
        
    def on_train_epoch_end(self, trainer, pl_module:LitMAE):
        
        print(f"Epoch {trainer.current_epoch} - Checking if KNN evaluation should be run...")
        print(f"Evaluation frequency is set to {self.eval_frequency}.")
        if trainer.current_epoch % self.eval_frequency != 0:
            print(f"Skipping KNN evaluation for epoch {trainer.current_epoch} as it is not a multiple of {self.eval_frequency}.")
            return
        print(f"Running KNN evaluation for epoch {trainer.current_epoch}...")
        
        
        # Perform evaluation here
        knn_model = KNNSegmentation(
            encoder=pl_module.get_encoder(),
            classes=self.classes,
            train_loader=self.train_loader,
            val_loader=self.val_dataloader,
        )

        mean_ious = knn_model.evaluate(k=3, distance_metric="cosine", weights="distance")
        mean_iou = sum(mean_ious.values()) / len(mean_ious)
        pl_module.log("val/knn_mean_iou", mean_iou)


class PatchEvaluationCallback(pl.Callback):
    def __init__(self, batch_size, eval_data_path, classes, num_workers, input_resolution, eval_frequency):

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
            normalize=config.normalize
        )
        
        self.eval_module.setup(stage="fit")
        self.train_loader = self.eval_module.train_dataloader()
        self.val_dataloader = self.eval_module.val_dataloader()
        self.classes = classes
        self.eval_frequency = eval_frequency
        
        
    def on_train_epoch_end(self, trainer, pl_module:LitMAE):
        
        print(f"Epoch {trainer.current_epoch} - Checking if KNN evaluation should be run...")
        print(f"Evaluation frequency is set to {self.eval_frequency}.")
        if trainer.current_epoch % self.eval_frequency != 0:
            print(f"Skipping KNN evaluation for epoch {trainer.current_epoch} as it is not a multiple of {self.eval_frequency}.")
            return
        print(f"Running KNN evaluation for epoch {trainer.current_epoch}...")
        
        
        # Perform evaluation here
        knn_model = KNNHummingBirdSegmentation(
            encoder=pl_module.get_encoder(),
            classes=self.classes,
            train_loader=self.train_loader,
            val_loader=self.val_dataloader,
            batch_size=config.test_batch_size
        )

        mean_ious = knn_model.evaluate(k=30, thresholds=[0.3]*len(self.classes), betas=0.02)
        iou_list = [m_iou for m_iou in mean_ious.values() if m_iou is not None]
        if not iou_list:
            print("No valid IoUs found. Skipping logging.")
            return
        mean_iou = sum(iou_list) / len(iou_list)
        pl_module.log("val/patch_retrieval_mean_iou", mean_iou)

def run_segmentation(
    config: argparse.Namespace,
):
    
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
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.seed)

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
        transform =A.Compose(transforms),
    )

        # Add this to both scripts after config creation
    print("=== CONFIG DEBUG ===")
    print(f"Tracking URI: {getattr(config, 'tracking_uri', 'NOT SET')}")
    print(f"Experiment name: {getattr(config, 'name', 'NOT SET')}")
    print(f"Run name: {getattr(config, 'run_name', 'NOT SET')}")
    print(f"Job ID: {getattr(config, 'job_id', 'NOT SET')}")

    loggers = utils.get_loggers(
        job_id=config.job_id,
        tracking_uri=config.tracking_uri,
        logging_path=str(config.path.absolute()),
        run_name=config.run_name,
        experiment_n=config.experiment_n,
        experiment_name=config.name,
        use_tb_logger=True,
    )

    if getattr(config, 'ckpt_path', None):
        model = LitMAE.load_from_checkpoint(config.ckpt_path, parameters=config.model_params, 
                                            model=MaskedAutoencoderFasterVit, strict=False)

    else:
        model = LitMAE(parameters=config.model_params, model=MaskedAutoencoderFasterVit)
        print("No checkpoint path provided, initializing model from scratch.")
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

    ckpt_callback_every_200 = ModelCheckpoint(
        dirpath=str(config.path),
        filename="epoch{epoch:04d}-periodic-save",
        every_n_epochs=200,
        save_last=False,
        save_top_k= -1,  # Optional: don't duplicate the last checkpoint
        save_on_train_epoch_end=True,
    )

    callbacks = [
        ckpt_callback,
        #ckpt_callback_every_200, # Uncomment to enable periodic saving
        EarlyStopping(
            patience=config.patience, monitor=config.monitor, mode="max" if config.monitor == "val/knn_mean_iou" else "min",
            verbose=True
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


    if config.model_params.get('accumulate_grad_batches'):
        warnings.warn(f"Accumulating gradients over {config.model_params['accumulate_grad_batches']} batches.")


    check_val_every_n_epoch = config.check_val_every_n_epoch if hasattr(config, 'check_val_every_n_epoch') else 1
    if "knn_eval_frequency" in vars(config):
        print(f"Using knn_eval_frequency: {config.knn_eval_frequency}")
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
        gradient_clip_val=None, # Disable automatic gradient clipping as we're using manual optimization
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_checkpointing=True,
        num_sanity_val_steps=2,
        precision=config.precision,
        sync_batchnorm=True,
        devices="auto",
        accelerator="gpu",
        # limit_train_batches=5,  # Only 5 batches per epoch
        # limit_val_batches=2,    # Only 2 batches for validation
    )
    trainer.fit(model=model, datamodule=datamodule, 
                #ckpt_path = config.ckpt_path if hasattr(config, "ckpt_path") else None,
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
        default="./configs/pretrain/pretrain_vit_mae_only_model_config.yml",
        help="Path to model specific YAML file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/pretrain/pretrain_main.yml",
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
        default=None
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

    parser.add_argument("--ckpt_path", 
        type=str,
        default=None,
        help="Path to model checkpoint"
    )

    parser.add_argument("--accumulate_grad_batches",
        type=int,
        help="Number of batches to accumulate before backprop. This is to support smaller patch sizes.")

    parser.add_argument("--vit_type",
        type=str,
        default=None,
        help="Type of ViT model to use",
    )

    parser.add_argument("--reconstruction_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the reconstruction loss",
    )

    parser.add_argument("--dino_loss_weight",
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
        default=None,)
    
    parser.add_argument(
        "--knn_eval_frequency",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--monitor",
        type=str,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # Initialize save directory
    args.path = Path(args.path)
    args.job_id = os.environ.get("SLURM_JOB_ID", "")

    # Create repository after merging name from args and config
    config = utils.merge_config(args.config, args, args.model_config)
    print("Config after merge: ", config)
    config.path = args.path / f"{config.name}_{config.job_id}"
    print(config.path)
    config.path.mkdir(exist_ok=True, parents=True)
    print("Is cuda available? ", torch.cuda.is_available())
    config.run_name = f"{config.job_id}-{config.run_key}"

    # Global data collection
    config.data_collection_path = (
        Path(config.data_collection_path)
        if "data_collection_path" in vars(config)
        else config.path / "experiment_data"
    )
    config.data_collection_path.mkdir(exist_ok=True, parents=True)

    print("Running pretraining with config:")
    print(config)

    run_segmentation(config)
