from typing import Any

import math
from collections.abc import Iterable

from torch import optim
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from segmentation.models.model import PLBaseModel
from segmentation.models.vit.vit_seg import ViTSegmentation
from segmentation.models.faster_vit.faster_vit_seg import FasterVitSeg
from segmentation.visualization import SegmentationLoggerMixin

from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss

from time import perf_counter


class LitViTSegmentation(PLBaseModel, SegmentationLoggerMixin):
    def __init__(
        self,
        parameters: dict,
        classes: list,
        model: ViTSegmentation | FasterVitSeg = ViTSegmentation,
    ):
        super().__init__(model=model, parameters=parameters, classes=classes)

        # Optimizer and scheduler
        self.lr = parameters["lr"]
        self.min_lr = parameters["min_lr"]
        self.weight_decay = parameters["weight_decay"]
        self.warmup_epochs = parameters.get("warmup_steps", 0)
        self.layer_decay = parameters.get("layer_decay", 0.75)
        self.use_annealing_lr = parameters.get("use_annealing_lr", True)
        self.use_decay_lr = parameters.get("use_decay_lr", True)
        self.model_params = parameters
        self.loss_name = parameters.get("loss", "bce")
        self.head_lr_multiplier = parameters.get("head_lr_multiplier", 10.0)
        # loss function
        if self.loss_name == "jaccard":
            print("Using Jaccard loss.")
            self.loss_fn = self.custom_jaccard_loss
        elif self.loss_name == "bce":
            print("Using Binary Cross Entropy loss.")
            self.loss_fn = self.custom_binary_cross_entropy
        elif self.loss_name == "dice":
            print("Using Dice loss.")
            self.loss_fn = self.custom_dice_loss
        
        
        # double view option
        self.double_view = parameters.get("double_view", False)
        print("Double view is in LitVitSeg: ", self.double_view)
        print("Successfully created LitVitSegmentation model.")

    def custom_jaccard_loss(self, y_hat: torch.Tensor, batch: dict):
        print("Class mask shape: ", batch["class_mask"].shape)
        print("Y_hat shape: ", y_hat.shape)

        class_mask = batch["class_mask"].moveaxis(-1, 1)
        loss_fn = JaccardLoss(
                mode="multilabel",
                classes=len(self.classes),
                from_logits=True,              
        )

        loss = loss_fn(
            y_hat, class_mask
        )

        return loss, y_hat.moveaxis(1, -1)

    def custom_dice_loss(self, y_hat: torch.Tensor, batch: dict):
        print("Class mask shape: ", batch["class_mask"].shape)
        print("Y_hat shape: ", y_hat.shape)

        class_mask = batch["class_mask"].moveaxis(-1, 1)
        loss_fn = DiceLoss(
            mode="multilabel",
            classes=len(self.classes),
            from_logits=True,
        )

        loss = loss_fn(
            y_hat, class_mask
        )

        return loss, y_hat.moveaxis(1, -1)

    def custom_cross_entropy(self, y_hat: torch.Tensor, batch: dict):
        print("Class mask shape: ", batch["class_mask"].shape)
        class_indices = torch.argmax(batch["class_mask"], dim=-1)  # Shape: (batch_size, height, width)
        print("Class indices shape: ", class_indices.shape)
        loss = self.loss_fn(
            y_hat, class_indices, reduction="none")
        
        weight_index = class_indices.unsqueeze(-1)
        selected_weights = batch["ignore_mask"][weight_index]
        weights = 1 - selected_weights
        
        print("Weights shape: ", weights.shape)
        loss = (loss * weights).mean()
        return loss, y_hat
    
    def custom_binary_cross_entropy(self, y_hat: torch.Tensor, batch: dict):
        y_hat = y_hat.moveaxis(1, -1)
        print("Y_hat shape: ", y_hat.shape)
        loss = F.binary_cross_entropy_with_logits(
            y_hat,
            batch["class_mask"],
            weight=(1 - batch["ignore_mask"]),
        )
        return loss, y_hat

    def training_step(self, batch: Any, batch_idx: int):  

        print("Arrived at training step!")
        
        if hasattr(self, "batch_start"):
            batch_time = perf_counter() - self.batch_start
            print(f"Time taken for previous batch: {batch_time:.4f} seconds") 

        start = perf_counter()
        # y_hat shape -> (batch_size, num_classes, height, width)
        # class_mask shape -> (batch_size, height, width, num_classes) 
        y_hat, _ = self.model(batch["image"])
        end = perf_counter()
        print(f"Time taken for forward pass: {end - start:.4f} seconds")

        if self.double_view:
            y_hat_first_half = y_hat[:y_hat.shape[0] // 2]
            y_hat_second_half = y_hat[y_hat.shape[0] // 2:]
            #y_hat = (y_hat_first_half + y_hat_second_half)/2

            #y_hat = F.sigmoid(y_hat)
            loss1, _ = self.loss_fn(
                y_hat, batch
            )
            loss2, _ = self.loss_fn(
                y_hat_first_half, batch
            )

            loss = (loss1 + loss2)/2
        else:
            #y_hat = F.sigmoid(y_hat)
            start_loss = perf_counter()
            loss, _ = self.loss_fn(
                y_hat, batch
            )
            end_loss = perf_counter()
            print(f"Time taken for loss computation: {end_loss - start_loss:.4f} seconds")

        # Log metrics
        self.log("train/loss", loss, batch_size=batch["image"].shape[0], sync_dist=True, on_epoch=True)
        self.batch_start = perf_counter()
        return loss

    def validation_step(self, batch: Any, batch_idx: int):

        y_hat, _ = self.model(batch["image"])

        if self.double_view:
            y_hat_first_half = y_hat[:y_hat.shape[0] // 2]
            y_hat_second_half = y_hat[y_hat.shape[0] // 2:]
            y_hat = (y_hat_first_half + y_hat_second_half)/2
        
        #y_hat = F.sigmoid(y_hat)
        #y_hat = y_hat.moveaxis(1, -1)
        loss, y_hat = self.loss_fn(y_hat, batch)

        # Log metrics (for now, mean iou and AUC of singular values of embeddings)
        with torch.no_grad():
            metrics = self.calculate_metrics(
                y_hat > 0.5, batch["class_mask"], batch["ignore_mask"]
            )

        self.log_dict({"val/loss": loss, **metrics})
        self.log_segmentation_overlay(
            self.trainer.loggers,
            batch["image"],
            y_hat > 0.5,
            batch_idx,
            context="val",
            epoch=self.current_epoch,
        )

        return loss

    def test_step(self, batch: Any, batch_idx: int):
        y_hat, _ = self.model(batch["image"])

        if self.double_view:
            y_hat_first_half = y_hat[:y_hat.shape[0] // 2]
            y_hat_second_half = y_hat[y_hat.shape[0] // 2:]
            y_hat = (y_hat_first_half + y_hat_second_half)/2

        #y_hat = F.sigmoid(y_hat)

        loss, y_hat = self.loss_fn(y_hat, batch)

        # Log metrics
        metrics = self.calculate_metrics(
            y_hat > 0.5,
            batch["class_mask"],
            batch["ignore_mask"],
            context="test",
        )
        self.log_dict({"test/loss": loss, **metrics})

        self.log_segmentation_overlay(
            self.trainer.loggers,
            batch["image"],
            y_hat > 0.5,
            batch_idx,
            filenames=batch["name"],
            context="test",
            epoch=self.current_epoch,
        )

        # Log MESA metrics if finetuning
        self.mesa_evaluation(
            batch,
            batch_idx,
            log_images_fn=self.log_segmentation_overlay,
            context="ema/test",
        )
        return loss

    def _initialize_optimizer_reversed_decay(self, decayed_lr: bool = True):
        """
        Initialize the optimizer with decayed learning rates for each layer.
        """
        if decayed_lr:
            print("Using decayed learning rate with layer decay (similar domain).")
            param_groups = []
            num_blocks = len(self.model.encoder.blocks)
            
            for i, block in enumerate(self.model.encoder.blocks):
                # REVERSED: Earlier blocks get HIGHER LR, later blocks get LOWER LR
                param_groups.append({
                    "params": block.parameters(),
                    "lr": self.lr * (self.layer_decay ** i)  # ← Changed: removed (num_blocks - i - 1)
                })

            # Patch embedding gets moderate LR (needs input channel adaptation)
            param_groups.append({
                "params": self.model.encoder.patch_embed.parameters(),
                "lr": self.lr * (self.layer_decay ** (num_blocks // 2))  # ← Moderate decay
            })

            # Progressive LR for FPN input blocks (shallower = higher LR)
            for i, block in enumerate(self.model.decoder.fpn_in):
                lr = self.lr * self.head_lr_multiplier * (1.0 - 0.15 * i)
                param_groups.append({
                    "params": block.parameters(),
                    "lr": lr,
                })

            # Progressive LR for FPN output blocks
            for i, block in enumerate(self.model.decoder.fpn_out):
                lr = self.lr * self.head_lr_multiplier * (0.8 - 0.1 * i)
                param_groups.append({
                    "params": block.parameters(),
                    "lr": lr,
                })

            # PPM module blocks (usually same LR for all)
            for block in self.model.decoder.ppm:
                param_groups.append({
                    "params": block.parameters(),
                    "lr": self.lr * self.head_lr_multiplier,
                })

            # PPM last conv
            param_groups.append({
                "params": self.model.decoder.ppm_last_conv.parameters(),
                "lr": self.lr * self.head_lr_multiplier,
            })

            # Fusion conv
            param_groups.append({
                "params": self.model.decoder.conv_fusion.parameters(),
                "lr": self.lr * self.head_lr_multiplier,
            })

            # Head (final classifier)
            param_groups.append({
                "params": self.model.decoder.head.parameters(),
                "lr": self.lr * self.head_lr_multiplier,
            })
            print("Parameter groups for optimizer:")
            for i, group in enumerate(param_groups):
                print(f"Group {i}: LR = {group['lr']}")
            
            optimizer = optim.AdamW(
                param_groups, weight_decay=self.weight_decay
            )
        else:
            print("Using same learning rate for all layers.")
            optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        return optimizer
    
    def _initialize_optimizer_normal(self, decayed_lr: bool = True):
        """
        Initialize the optimizer with decayed learning rates for each layer.
        """
        if decayed_lr:
            print("Using decayed learning rate with layer decay.")
            param_groups = []
            num_blocks = len(self.model.encoder.blocks)
            print(f"Number of blocks in encoder: {num_blocks}")

            for i, block in enumerate(self.model.encoder.blocks):
                param_groups.append({
                    "params": block.parameters(),
                    "lr": self.lr * (self.layer_decay ** (num_blocks - i - 1))
                })

            # Patch embedding layer receives the learning rate with highest decay
            param_groups.append({
                "params": self.model.encoder.patch_embed.parameters(),
                "lr": self.lr * (self.layer_decay ** num_blocks)
            })

            # Don't decay the parameters of the decoder as we're training it from scratch
            param_groups.append({"params": self.model.decoder.parameters(), "lr": self.lr})

            print("Parameter groups for optimizer:")
            for i, group in enumerate(param_groups):
                print(f"Group {i}: LR = {group['lr']}")
            
            optimizer = optim.AdamW(
                param_groups, weight_decay=self.weight_decay
            )
        else:
            print("Using same learning rate for all layers.")
            optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        return optimizer

    def _initialize_optimizer(self, decayed_lr: bool = True):
        """
        Initialize the optimizer based on the layer decay setting.
        """
        if self.model_params.get("use_reversed_decay", False):
            print("Using reversed decay.")
            return self._initialize_optimizer_reversed_decay(decayed_lr=decayed_lr)
        else:
            print("Using standard optimizer settings.")
            return self._initialize_optimizer_normal(decayed_lr=decayed_lr)

    def _initialize_annealed_lr(self, decayed_lr: bool = True):

        optimizer = self._initialize_optimizer(decayed_lr=decayed_lr)

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
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def _initialize_constant_lr(self, decayed_lr: bool = False):
        optimizer = self._initialize_optimizer(decayed_lr)
        return {"optimizer": optimizer}

    def configure_optimizers(self):
        if self.use_annealing_lr:
            print("Using cosine annealing learning rate scheduler.")
            return self._initialize_annealed_lr(decayed_lr=self.use_decay_lr)
        else:
            print("Using constant learning rate.")
            return self._initialize_constant_lr(decayed_lr=self.use_decay_lr)


