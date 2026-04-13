from typing import Any, Optional, Callable
from abc import ABC, abstractmethod

import torch
import torchmetrics
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.wrappers import ClasswiseWrapper

from data.image_tiling import reconstruct_image_and_prediction
from segmentation.metrics import Metrics

# Create a binary metric for this single class
from torchmetrics import JaccardIndex


class SafeJaccardIndex:
    def __init__(self, jaccard_metric):
        self.jaccard_metric = jaccard_metric

    def __call__(self, preds, target):
        # For multi-label case, check each class individually
        result = {}
        for i, key in enumerate(self.jaccard_metric.labels):
            # Sum over all dimensions except the class dimension
            # target shape: (batch, classes, height, width)
            class_positive_count = target[:, i].sum()

            prefixed_key = f"iou_{key}"

            if class_positive_count == 0:
                result[prefixed_key] = None
            else:
                # Extract single class data and compute IoU
                class_preds = preds[:, i].float()  # Keep dimension
                class_target = target[:, i].float()  # Keep dimension

                single_class_metric = JaccardIndex(task="binary").to(preds.device)
                result[prefixed_key] = single_class_metric(
                    class_preds, class_target
                ).item()

        return result


class PLBaseModel(pl.LightningModule, ABC):
    def __init__(self, model, parameters, classes=None):
        super().__init__()
        self.classes = classes
        self.model = model(**parameters)

        parameters["model"] = self.model.__class__.__name__
        parameters["num_param"] = sum(p.numel() for p in self.model.parameters())

        self.save_hyperparameters(parameters)

        # Check if classes are provided for segmentation
        if classes:
            # Metrics
            self.iou_classwise = SafeJaccardIndex(
                ClasswiseWrapper(
                    JaccardIndex(
                        task="multilabel",
                        num_labels=len(self.classes),
                        average="none",
                    ),
                    labels=self.classes,
                    prefix="iou_",
                )
            )

            self.ap_classwise = ClasswiseWrapper(
                torchmetrics.AveragePrecision(
                    task="multilabel",
                    num_labels=len(self.classes),
                    average="none",
                ),
                labels=classes,
                prefix="ap_",
            )

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def tiled_inference(self, batch: Any, use_ema=False):
        """Infer over a grid of image tiles and stitch them back together
        before returning the prediction and adjusted image in the batch.

        Args:
            batch (Any): A dictionary of training samples

        Returns:
            tuple(torch.Tensor, Any): the prediction and batch
        """
        y_hat = []
        imgs = []

        for idx in range(batch["image"].size(0)):
            tile_H, tile_W, C, H, W = batch["image"][idx].shape

            # Inference reshaping the tiles to tile_H x tile_W
            if use_ema:
                pred = self.ema_model.module(
                    batch["image"][idx].reshape(-1, C, H, W).data.detach()
                )
            else:
                pred = self.model(batch["image"][idx].reshape(-1, C, H, W))
            pred = pred.moveaxis(1, -1).reshape(tile_H, tile_W, H, W, len(self.classes))

            # Reconstruct the original image and prediction shapes
            pred, img = reconstruct_image_and_prediction(
                img_tiles=batch["image"][idx],
                y_hat_tiles=pred,
                original_size=(
                    batch["class_mask"][idx].size(0),
                    batch["class_mask"][idx].size(1),
                ),
                overlap=batch["tile_overlap"][idx],
            )
            y_hat.append(pred)
            imgs.append(img)

        y_hat = torch.stack(y_hat)
        batch["image"] = torch.stack(imgs)
        return y_hat, batch

    def mesa_evaluation(
        self,
        batch: Any,
        batch_idx: int,
        context: str = "ema/val",
        log_images_fn: Optional[Callable] = None,
    ) -> dict | None:
        """Evaluates the batch on the MESA EMA model and logs the results directly

        Args:
            batch (Any): The current batch
            batch_idx (int): Index of the current batch
            context (str, optional): Context for metric dict
            variables. Defaults to "val/ema/".
        """
        if not self.mesa_finetune:
            return None

        # Handle inference for tiling
        if torch.all(batch["is_tiled"]):
            # Stores the batch predictions
            y_ema, batch = self.tiled_inference(batch, use_ema=True)
        else:
            # Otherwise regular inference
            y_ema = self.model(batch["image"])
            y_ema = y_ema.moveaxis(1, -1)

        y_ema = F.sigmoid(y_ema.to(self.device)) > 0.5

        iou_dict = self.calculate_metrics(
            y_ema, batch["class_mask"], batch["ignore_mask"], context=context
        )
        self.log_dict(iou_dict)

        if log_images_fn:
            log_images_fn(
                self.trainer.loggers,
                batch["image"],
                y_ema,
                batch_idx,
                filenames=batch["name"],
                context=context,
                epoch=self.current_epoch,
            )

    def mesa_step(
        self, y_hat: torch.Tensor, loss: torch.Tensor, batch: Any
    ) -> torch.Tensor:
        """Do a MESA optimization step if enabled in the model configuration

        Args:
            y_hat (torch.Tensor): Predictions on the batch
            loss (torch.Tensor): Loss of the model on the batch
            batch (Any): The current batch

        Returns:
            torch.Tensor: Adjusted loss
        """

        if not self.mesa_finetune:
            return loss

        with torch.no_grad():
            y_ema = self.ema_model.module(batch["image"].data.detach())
            y_ema = y_ema.moveaxis(1, -1)
            kl_div = self.kl_loss(y_hat, y_ema.to(self.device))

        self.ema_model.update(self.model)
        loss += self.mesa_lmbda * kl_div
        return loss

    def calculate_metrics(
        self,
        y_hat: torch.Tensor,
        class_mask: torch.Tensor,
        ignore_mask: torch.Tensor | None,
        context: str = "val",
        threshold: float = 0.5,
        iou_threshold: float = False,
        embeddings: torch.Tensor = None,
    ):
        """Calculate the IoU, AP metrics for predictions, skips classwise if input is
            0d-tensor.

        Args:
            y_hat (torch.Tensor): Prediction of the model before thresholding!
            class_mask (torch.Tensor): The segmentation mask
            ignore_mask (torch.Tensor): The ignore mask
            context (str): The prepend for the return dict. defaults to `val`.
            threshold (float): Prediction threshold to determine class per pixel.
            iou_threshold (float): The IoU threshold above which we calculate AP metrics.
                If None calculate metrics at any threshold otherwise only at specified
                threshold. Defaults to None.
            embeddings (torch.Tensor): The embeddings of the input. If included collapse metrics are calculated.

        Returns:
            dict: The mean IoU, classwise IoUs, and optionally AUC & entropy of embeddings.
        """
        # Evaluate on the full image: do not mask-out any pixels based on auxiliary ignore/evaluable masks.
        _ = ignore_mask

        metric_dict = {}

        # Set ignore region labels and predictions to 0.
        if len(y_hat.size()) != 0:
            # Reshape the predictions and masks to allow for classwise IoU computation,
            # view does not work as they're not located contiguous.
            B, H, W, C = class_mask.shape  # one channel per class
            y_hat = y_hat.reshape(B * H * W, C).long()
            class_mask = class_mask.reshape(B * H * W, C).long()

            # consider that IoU doesn't take into account true negatives, so it can be 0 either because there are
            # no positives or because the model is not able to predict any positive class.
            iou_class = self.iou_classwise(y_hat, class_mask)

            # Extract valid IoU values for mean calculation
            valid_ious = []
            processed_iou_class = {}

            for k, v in iou_class.items():
                if v is not None:
                    # Handle both tensor and scalar values
                    scalar_val = v.item() if isinstance(v, torch.Tensor) else v
                    valid_ious.append(scalar_val)
                    processed_iou_class[f"{context}/{k}"] = scalar_val

            # Calculate mean IoU
            if valid_ious:
                mean_iou = torch.tensor(valid_ious).mean()

            iou_class = processed_iou_class

            # Remove classes not present in the classwise IoU computation
            # this drives the average down.

            # Compute collapse metrics if embeddings are provided
            if (embeddings is not None) and (len(embeddings.size()) > 0):
                mets = Metrics(embeddings)
                dim_collapse = mets.auc_embedding_collapse()
                dim_collapse_entropy = mets.entropy_embedding_collapse()

                if valid_ious:
                    metric_dict = {
                        **iou_class,
                        f"{context}/iou": mean_iou,
                        f"{context}/auc_embedding": dim_collapse,
                        f"{context}/entropy_embedding": dim_collapse_entropy,
                    }
                else:
                    metric_dict = {
                        # No valid classes to evaluate
                        f"{context}/auc_embedding": dim_collapse,
                        f"{context}/entropy_embedding": dim_collapse_entropy,
                    }
            else:
                if valid_ious:
                    metric_dict = {**iou_class, f"{context}/iou": mean_iou}
                else:
                    metric_dict = {
                        # No valid classes to evaluate
                    }

        print(f"Metrics for {context}: {metric_dict}")
        return metric_dict
