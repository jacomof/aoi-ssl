from typing import Any

import torch
import lightning.pytorch as pl
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR

from pretrain.ibot.ibot import iBot
from segmentation.models.model import PLBaseModel
from segmentation.metrics import Metrics
from pretrain.mae.lit_mae import LitMAE
import numpy as np


def compute_entropy(dino_probs):
    """
    Compute the entropy of the DINO probabilities.
    """
    # Convert to numpy array
    dino_probs = dino_probs

    assert torch.allclose(dino_probs.sum(dim=1), torch.ones(dino_probs.size(0), device=dino_probs.device), atol=1e-4), "Probabilities do not sum to 1"

    # Compute the entropy
    entropy = -torch.sum(dino_probs * torch.log(dino_probs + 1e-10), dim=1)
    entropy = torch.mean(entropy, dim=0)

    return entropy.item()

class LitiBot(PLBaseModel):
    def __init__(
        self,
        parameters: dict,
    ):

        super().__init__(model=iBot, parameters=parameters)

        self.automatic_optimization=False
        self.warmup_epochs = parameters.get("warmup_epochs", 0)
        self.lr = float(parameters["lr"])
        print(f"Learning rate: {self.lr}")
        self.min_lr = float(parameters.get("min_lr", 0))
        self.accumulate_grad_batches = parameters.get("accumulate_grad_batches", 1)



    def training_step(self, batch: Any, batch_idx: int):
        x = batch["image"]
        masks = batch["mask"]

        batch_size = x[0].size(0)
        loss, mim_loss, dino_loss, teacher_embs, student_cls_mean, student_embs_avg_for_log \
            = self.model(x, masks)
        
        # Gradient accumulation logic
        optimizer = self.optimizers()
        self.manual_backward(loss)

        # Only step optimizer every accumulate_grad_batches steps
        if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (
            (batch_idx + 1) == self.trainer.num_training_batches
        ):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            self.model.teacher_ema.update(self.model.encoder)


        teacher_entropy = compute_entropy(teacher_embs)
        student_entropy = compute_entropy(student_embs_avg_for_log)

        self.log("train/loss", loss, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("train/mim_loss", mim_loss, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("train/dino_loss", dino_loss, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("train/teacher_entropy", teacher_entropy, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("train/student_cls_mean", student_cls_mean, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("train/student_entropy", student_entropy, sync_dist=True, batch_size=batch_size, on_epoch=True)

        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):

        x = batch["image"]
        masks = batch["mask"]

        batch_size = x[0].size(0)

        loss, mim_loss, dino_loss, teacher_embs, student_cls_mean, student_embs_avg_for_log = self.model(
            x, masks
        )

        teacher_entropy = compute_entropy(teacher_embs)
        student_entropy = compute_entropy(student_embs_avg_for_log)

        print(f"Validation step {batch_idx} - Loss: {loss.item()}")
        self.log("val/loss", loss, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("val/mim_loss", mim_loss, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("val/dino_loss", dino_loss, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("val/teacher_entropy", teacher_entropy, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("val/student_cls_mean", student_cls_mean, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log("val/student_entropy", student_entropy, sync_dist=True, batch_size=batch_size, on_epoch=True)
        return {'loss': loss}
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch["image"]
        masks = batch["mask"]

        batch_size = x.size(0)

        loss, _, _, teacher_embs, _, _ = self.model(
            x, masks
        )

        return {'loss': loss}

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=0.05
        )
        warmup_scheduler = LambdaLR(optimizer, lambda epoch: min(1.0, (epoch + 1) / self.warmup_epochs))
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - self.warmup_epochs,
                                             eta_min=self.min_lr)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
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