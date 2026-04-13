from typing import Any

from torch import optim
import torch.nn.functional as F

from segmentation.models.model import PLBaseModel
from segmentation.visualization import SegmentationLoggerMixin
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch
import segmentation_models_pytorch as smp


class LitDeeplabSegmentation(PLBaseModel, SegmentationLoggerMixin):
    def __init__(
        self, parameters: dict, classes: list, model: smp.DeepLabV3 = smp.DeepLabV3
    ):
        super().__init__(model=model, parameters=parameters, classes=classes)

        pretrained_resnet = bool(parameters.get("pretrained", False))
        num_classes = (
            len(classes) if classes is not None else parameters.get("num_classes")
        )
        self.model = smp.DeepLabV3(
            encoder_name="resnet18",
            encoder_weights=(
                "imagenet" if pretrained_resnet else None
            ),  # or "imagenet" if you want pretrained weights
            in_channels=2,
            classes=num_classes,
            activation=None,  # We'll apply sigmoid in your existing code
        )

        # Optimizer and scheduler
        self.lr = float(parameters["lr"])
        self.warmup_epochs = int(parameters["warmup_epochs"])
        self.min_lr = float(parameters["min_lr"])
        self.use_annealing_lr = bool(parameters["use_annealing_lr"])
        self.use_decay_lr = bool(parameters["use_decay_lr"])
        self.weight_decay = float(parameters.get("weight_decay", 0.0))

    def training_step(self, batch: Any, batch_idx: int):
        y_hat = self.model(batch["image"])
        y_hat = y_hat.moveaxis(1, -1)

        loss = F.binary_cross_entropy_with_logits(
            y_hat, batch["class_mask"], weight=(1 - batch["ignore_mask"])
        )

        # Log metrics
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        y_hat = self.model(batch["image"])
        y_hat = y_hat.moveaxis(1, -1)

        loss = F.binary_cross_entropy_with_logits(
            y_hat, batch["class_mask"], weight=(1 - batch["ignore_mask"])
        )

        y_hat_probs = F.sigmoid(y_hat)

        # Log metrics
        metrics = self.calculate_metrics(
            y_hat_probs > 0.5,
            batch["class_mask"],
            batch["ignore_mask"],
        )
        self.log_dict({"val/loss": loss, **metrics})

        return loss

    def test_step(self, batch: Any, batch_idx: int):
        y_hat = self.model(batch["image"])
        y_hat = y_hat.moveaxis(1, -1)

        loss = F.binary_cross_entropy_with_logits(
            y_hat, batch["class_mask"], weight=(1 - batch["ignore_mask"])
        )

        y_hat_probs = F.sigmoid(y_hat)

        # Log metrics
        metrics = self.calculate_metrics(
            y_hat_probs > 0.5,
            batch["class_mask"],
            batch["ignore_mask"],
            context="test",
        )
        self.log_dict({"test/loss": loss, **metrics})

        return loss

    def _initialize_optimizer(self, decayed_lr: bool = True):
        """Configure optimizer with layer-wise learning rate decay for SMP DeepLabV3"""

        if not decayed_lr:
            return optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        base_lr = self.lr
        decay_rate = getattr(self, "layer_lr_decay", 0.75)
        head_multiplier = getattr(self, "head_lr_multiplier", 10)

        # Group parameters by layer depth
        param_groups = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine learning rate based on SMP structure
            if "encoder" in name:
                # SMP encoder structure: encoder.layer1, encoder.layer2, etc.
                if any(layer in name for layer in ["conv1", "bn1"]):
                    lr = base_lr  # Stem layers (highest LR for encoder)
                elif "layer1" in name:
                    lr = base_lr * (decay_rate**1)
                elif "layer2" in name:
                    lr = base_lr * (decay_rate**2)
                elif "layer3" in name:
                    lr = base_lr * (decay_rate**3)
                elif "layer4" in name:
                    lr = base_lr * (decay_rate**4)
                else:
                    lr = base_lr * (decay_rate**2)  # Default encoder layers
            elif "decoder" in name:
                # SMP decoder (ASPP module in DeepLabV3)
                lr = base_lr * head_multiplier * 0.5  # Slightly lower than head
            elif "segmentation_head" in name:
                # Final segmentation head gets highest learning rate
                lr = base_lr * head_multiplier
            elif "classification_head" in name:
                # Classification head (if present)
                lr = base_lr * head_multiplier
            else:
                # Any other parameters (batch norm, etc.)
                lr = base_lr

            param_groups.append(
                {
                    "params": [param],
                    "lr": lr,
                    "weight_decay": (
                        self.weight_decay
                        if "bn" not in name and "bias" not in name
                        else 0.0
                    ),
                }
            )

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def _initialize_annealed_lr(self, decayed_lr: bool = True):

        optimizer = self._initialize_optimizer(decayed_lr)

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

    def _initialize_constant_lr(self, decayed_lr: bool = True):
        optimizer = self._initialize_optimizer(decayed_lr)
        return {"optimizer": optimizer}

    def configure_optimizers(self):
        if self.use_annealing_lr:
            return self._initialize_annealed_lr(decayed_lr=self.use_decay_lr)
        else:
            return self._initialize_constant_lr(decayed_lr=self.use_decay_lr)
