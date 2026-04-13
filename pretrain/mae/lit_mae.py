# MAE lighting module

from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from .mae import MaskedAutoencoder
from ..dino.mae_dino import DinoMaskedAutoencoder
from .mae_fastervit import MaskedAutoencoderFasterVit
from segmentation.models.model import PLBaseModel
from segmentation.metrics import Metrics


class LitMAE(PLBaseModel):
    def __init__(
        self,
        parameters: dict,
        model: (
            MaskedAutoencoder | DinoMaskedAutoencoder | MaskedAutoencoderFasterVit
        ) = MaskedAutoencoder,
        current_epoch: int = None,
        ckpt_path: str = None,
    ):
        super().__init__(model=model, parameters=parameters)

        self.lr = float(parameters["lr"])
        self.min_lr = float(parameters.get("min_lr", 0))

        # Name discrepancy to make the configs more reusable
        self.warmup_epochs = parameters.get("warmup_steps")

        self.mask_ratio = parameters.get("mask_ratio")

        self.custom_current_epoch = current_epoch
        self.ckpt_path = ckpt_path
        self.custom_loading = ckpt_path is not None and current_epoch is not None

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["image"]
        batch_size = x.size(0)

        loss, _, _, patch_embed = self.model(
            x, mask_ratio=self.mask_ratio, compute_loss=True
        )

        # Uncomment to compute metrics, but this will slow down training
        # with torch.no_grad():
        #     mets = Metrics(patch_embed)
        #     auc = mets.auc_embedding_collapse()
        #     entropy = mets.entropy_embedding_collapse()
        # self.log("train/dim_collapse_auc", auc, sync_dist=True, batch_size=batch_size, on_step=False, on_epoch=True)
        # self.log("train/dim_collapse_entropy", entropy, sync_dist=True, batch_size=batch_size, on_step=False, on_epoch=True)

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
        )
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["image"]
        batch_size = x.size(0)

        loss, _, _, patch_embeds = self.model(
            x, mask_ratio=self.mask_ratio, compute_loss=True
        )

        # Uncomment to compute metrics, but this will slow down training
        # mets = Metrics(patch_embeds)
        # auc = mets.auc_embedding_collapse()
        # entropy = mets.entropy_embedding_collapse()
        # self.log('val/dim_collapse_auc', auc, batch_size=batch_size, on_step=True, on_epoch=True)
        # self.log('val/dim_collapse_entropy', entropy, batch_size=batch_size, on_step=True, on_epoch=True)

        self.log("val/loss", loss, batch_size=batch_size, on_step=True, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch["image"]
        batch_size = x.size(0)

        loss, _, _, patch_embeds = self.model(
            x, mask_ratio=self.mask_ratio, compute_loss=True
        )
        mets = Metrics(patch_embeds)
        auc = mets.auc_embedding_collapse()
        entropy = mets.entropy_embedding_collapse()

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/dim_collapse_auc", auc, batch_size=batch_size)
        self.log("test/dim_collapse_entropy", entropy, batch_size=batch_size)

        return {"loss": loss}

    def on_train_start(self):

        if self.custom_loading:
            # Get the current epoch from the trainer
            current_epoch = self.trainer.current_epoch
            # Get the scheduler from the trainer
            scheduler = self.trainer.lr_scheduler_configs[0].scheduler
            # Advance the scheduler to the correct epoch
            for _ in range(current_epoch):
                scheduler.step()

            if self.ckpt_path is not None:
                checkpoint = torch.load(
                    self.ckpt_path, map_location="cpu", weights_only=False
                )
                if (
                    "optimizer_states" in checkpoint
                    and len(self.trainer.optimizers) > 0
                ):
                    optimizer_state = checkpoint["optimizer_states"][0]
                    self.trainer.optimizers[0].load_state_dict(optimizer_state)

            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())

            self.trainer.current_epoch = self.custom_current_epoch
            self.trainer.global_step = (
                self.custom_current_epoch * steps_per_epoch
            )  # If you know steps_per_epoch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=0.05
        )
        warmup_scheduler = LambdaLR(
            optimizer, lambda epoch: min(1.0, (epoch + 1) / self.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.warmup_epochs,
            eta_min=self.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_encoder(self):
        return self.model.encoder
