# DINO lighting module

from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR

from pretrain.dino.mae_dino import DinoMaskedAutoencoder
from segmentation.metrics import Metrics
from pretrain.mae.lit_mae import LitMAE


def compute_entropy(dino_probs):
    """
    Compute the entropy of the DINO probabilities.
    """
    # Convert to numpy array
    dino_probs = dino_probs

    # Compute the entropy
    entropy = -torch.sum(dino_probs * torch.log(dino_probs + 1e-10), axis=1)
    entropy = torch.mean(entropy, axis=0)

    return entropy.item()


class LitMAEDino(LitMAE):
    def __init__(
        self,
        parameters: dict,
    ):

        super().__init__(model=DinoMaskedAutoencoder, parameters=parameters)

        self.mask_ratio = parameters["mask_ratio"]
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.automatic_optimization = False
        # Name discrepancy to make the configs more reusable
        self.warmup_epochs = parameters["warmup_epochs"]

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["image"]

        batch_size = x[0].size(0)
        loss, cls_emb_student, cls_emb_teacher = self.model(
            x, mask_ratio=self.mask_ratio
        )

        # Uncomment the following lines to check memory usage
        # memory_usage = torch.cuda.memory_allocated() / (1024 ** 3)
        # print(f"Memory usage: {memory_usage:.2f} GB")

        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss)  # Backward pass
        optimizer.step()  # Optimizer step
        optimizer.zero_grad(set_to_none=True)  # Zero gradients

        # Uncomment the following lines to check memory usage after optimizer step
        # print("Optimizer step successful")
        # memory_usage_after = torch.cuda.memory_allocated() / (1024 ** 3)
        # print(f"Memory usage after step: {memory_usage_after:.2f} GB")

        with torch.no_grad():
            self.model.teacher_ema.update(self.model.encoder)

        with torch.no_grad():
            entropy_teacher = compute_entropy(cls_emb_teacher)
            self.log(
                "train/dim_collapse_entropy_teacher",
                entropy_teacher,
                sync_dist=True,
                batch_size=batch_size,
                on_epoch=True,
            )

        self.log(
            "train/loss", loss, sync_dist=True, batch_size=batch_size, on_epoch=True
        )

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):

        x = batch["image"]

        batch_size = x[0].size(0)

        loss, cls_emb_student, cls_emb_teacher = self.model(
            x, mask_ratio=self.mask_ratio
        )

        with torch.no_grad():
            entropy_teacher = compute_entropy(cls_emb_teacher)
            self.log(
                "val/dim_collapse_entropy_teacher",
                entropy_teacher,
                sync_dist=True,
                batch_size=batch_size,
            )

        self.log("val/loss", loss, batch_size=batch_size, sync_dist=True)

        return {"loss": loss}

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch["image"]

        batch_size = x.size(0)

        loss, patch_embeds = self.model(
            x, mask_ratio=self.mask_ratio, compute_loss=True
        )
        mets = Metrics(patch_embeds)
        auc = mets.auc_embedding_collapse()
        entropy = mets.entropy_embedding_collapse()

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/dim_collapse_auc", auc, batch_size=batch_size)
        self.log("test/dim_collapse_entropy", entropy, batch_size=batch_size)

        return {"loss": loss}

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
                # this is not important as we use manual optimization
                # frequency is determined by how often the scheduler is called
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def get_encoder(self):
        return self.model.encoder
