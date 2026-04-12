from typing import Optional

import cv2
import torch
import pylab
import numpy as np

# from aimstack.base import Image
from lightning.pytorch.loggers import Logger
from lightning.pytorch import loggers as pl_loggers
# from aimstack.pytorch_lightning_tracker.loggers import BaseLogger as AimLogger


def overlay_segmentation(
    img: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.4,
    contour_thickness: int = 1,
    light_condition: int = 0,
    output_size: tuple[int, int] = (768, 768)
) -> np.ndarray:
    """Overlay the image with the segmentation predictions

    Args:
        img (torch.Tensor): The target image
        mask (torch.Tensor): The prediction mask
        alpha (float, optional): Transparency. Defaults to 0.4.
        contour_thickness (int, optional): thickness of the contour labels. Defaults to 1.
        light_condition (int, optional): Which light condition to overlay. Defaults to 0.
        output_size (tuple[int, int], optional): The image output size of the visualization.
            Defaults to (768, 768).

    Returns:
        np.ndarray: Visualization of the overlay
    """

    # Ignore the second lighting condition,
    img = (
        img[light_condition].clone().squeeze().detach().cpu().numpy() * 255
    ).astype(np.uint8)
    mask = mask.clone().squeeze().cpu().numpy().astype(np.uint8)

    n_colour = mask.shape[-1]
    # Gets colors for each class from the HSV colormap
    colors = [
        [int(255 * c) for c in pylab.get_cmap("hsv")(1.0 * i / n_colour)[:3]]
        + [int(255 * alpha)]
        for i in range(n_colour)
    ]

    # Initialize an RGBA array for each class
    lbl_rgba = [
        np.zeros((*mask.shape[:-1], 4), dtype=np.uint8) for _ in range(n_colour)
    ]

    # Create a mask for each class and draw contours
    for cls in range(n_colour):
        c_mask = mask[..., cls].astype(np.uint8)
        c = colors[cls]

        lbl_rgba[cls][c_mask > 0] = (*c[:-1], int(c[-1] * 0.5))
        cnt, _ = cv2.findContours(c_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(lbl_rgba[cls], cnt, -1, c, contour_thickness)

    mask = mask.astype(np.int32)
    num_labels = np.sum(mask, axis=-1)[..., np.newaxis]
    mix_colors = np.sum(np.array(lbl_rgba), axis=0)
    viz = np.where(num_labels > 1, mix_colors / num_labels, mix_colors)

    # Returns an RGB array with overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    viz = cv2.convertScaleAbs(viz)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGBA2RGB)
    viz = cv2.addWeighted(img_rgb, alpha, viz, 0.4, 0).astype(np.uint8)
    viz = cv2.resize(viz, output_size)

    # Output shape: (H, W, C)
    return viz

def overlay_segmentation_with_legend(
    img: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.4,
    contour_thickness: int = 1,
    light_condition: int = 0,
    output_size: tuple[int, int] = (768, 768),
    label_names: Optional[list[str]] = ["Wire", "Ball", "Wedge", "Epoxy"]
) -> np.ndarray:
    """Overlay the image with the segmentation predictions and add a legend bar.

    Args:
        img (torch.Tensor): The target image
        mask (torch.Tensor): The prediction mask
        alpha (float, optional): Transparency. Defaults to 0.4.
        contour_thickness (int, optional): thickness of the contour labels. Defaults to 1.
        light_condition (int, optional): Which light condition to overlay. Defaults to 0.
        output_size (tuple[int, int], optional): The image output size of the visualization.
            Defaults to (768, 768).
        label_names (Optional[list[str]]): List of class names.

    Returns:
        np.ndarray: Visualization of the overlay with legend
    """

    # Ignore the second lighting condition,
    img = (
        img[light_condition].clone().squeeze().detach().cpu().numpy() * 255
    ).astype(np.uint8)
    mask = mask.clone().squeeze().cpu().numpy().astype(np.uint8)

    n_colour = mask.shape[-1]
    # Gets colors for each class from the HSV colormap
    colors = [
        [int(255 * c) for c in pylab.get_cmap("hsv")(1.0 * i / n_colour)[:3]]
        + [int(255 * alpha)]
        for i in range(n_colour)
    ]

    # Initialize an RGBA array for each class
    lbl_rgba = [
        np.zeros((*mask.shape[:-1], 4), dtype=np.uint8) for _ in range(n_colour)
    ]

    # Create a mask for each class and draw contours
    for cls in range(n_colour):
        c_mask = mask[..., cls].astype(np.uint8)
        c = colors[cls]

        lbl_rgba[cls][c_mask > 0] = (*c[:-1], int(c[-1] * 0.5))
        cnt, _ = cv2.findContours(c_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(lbl_rgba[cls], cnt, -1, c, contour_thickness)

    mask = mask.astype(np.int32)
    num_labels = np.sum(mask, axis=-1)[..., np.newaxis]
    mix_colors = np.sum(np.array(lbl_rgba), axis=0)
    viz = np.where(num_labels > 1, mix_colors / num_labels, mix_colors)

    # Returns an RGB array with overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    viz = cv2.convertScaleAbs(viz)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGBA2RGB)
    viz = cv2.addWeighted(img_rgb, alpha, viz, 0.4, 0).astype(np.uint8)
    viz = cv2.resize(viz, output_size)

    # --- Add legend bar ---
    legend_height = 40
    legend = np.ones((legend_height, output_size[0], 3), dtype=np.uint8) * 255  # white bar

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    box_w = 40
    box_h = 30
    spacing = 10
    x_offset = spacing

    for i, name in enumerate(label_names):
        color = colors[i][:3]
        # Draw color box
        cv2.rectangle(legend, (x_offset, 5), (x_offset + box_w, 5 + box_h), color, -1)
        # Draw label text
        cv2.putText(
            legend,
            name,
            (x_offset + box_w + 5, 5 + box_h - 7),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        x_offset += box_w + spacing + 80  # Adjust spacing for next label

    # Concatenate legend bar to bottom of image
    viz_with_legend = np.concatenate([viz, legend], axis=0)

    # Output shape: (H+legend_height, W, C)
    return viz_with_legend

class SegmentationLoggerMixin:
    def log_segmentation_overlay(
        self,
        loggers: list[Logger],
        images: torch.Tensor,
        masks: torch.Tensor,
        batch_idx: int,
        epoch: int,
        context: str = "val",
        filenames: Optional[list[str]] = None,
    ) -> None:
        """Log the images with segmentation overlay to Tensorboard, or Aim using the Trainer
        loggers of PyTorch Lightning.

        Args:
            loggers (list[Logger]): List of PyTorch Lightning Loggers
            images (torch.Tensor): Batch of normalized training images
            masks (torch.Tensor): Batch of segmentation labels
            batch_idx (int): Batch index
            epoch (int): The current epoch
            context (str, optional): Current training, validation, test split. Defaults to "val".
            filenames (Optional[list[str]], optional): The corresponding image filenames.
            Defaults to None.

        Raises:
            ValueError: When no Tensorboard or Aim logger is passed.
        """

        viz_experiments = []
        for logger in loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                viz_experiments.append(logger.experiment)

        if not len(viz_experiments):
            raise ValueError("TensorBoard or Aim Logger not found")

        # Log the images (Give them different names)
        overlays = [
            overlay_segmentation(image, mask)
            for image, mask in zip(images, masks)
        ]

        for experiment in viz_experiments:
            caption = f"Image/overlay_{batch_idx}"

            for idx, overlay in enumerate(overlays):
                if filenames:
                    # Remove the file extension by splitting
                    caption = f"Image/{filenames[idx].split('.')[0]}"

                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    experiment.add_image(
                        caption,
                        overlay,
                        global_step=epoch,
                        dataformats="HWC",
                    )

                # if isinstance(logger, AimLogger):
                #     experiment.track(
                #         Image(overlay),
                #         name=caption,
                #         epoch=epoch,
                #         context=context,
                #     )
