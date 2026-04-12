import torch
import numpy as np
import torch.nn.functional as F


def slice_image_to_tiles(
    image: torch.Tensor, tile_size: int, overlap: int
) -> torch.Tensor:
    """Slicing an image of (C, H, W) to tiles of (tile_size, tile_size) having an
    overlap.

    Args:
        image (torch.Tensor): An image of (C, H, W).
        tile_size (int): Size of the square tile (tile_size, tile_size).
        overlap (int): Amount of overlap of the tiles.

    Returns:
        torch.Tensor: Returns the tiles as (tile_H, tile_W, C, tile_size, tile_size)
    """
    step = tile_size - overlap

    # Calculate necessary padding
    _, H, W = image.shape
    pad_h = (H - overlap) % step
    pad_w = (W - overlap) % step

    # Adjust padding to cover the whole image, missing
    # part which needs to be padded.
    if pad_h > 0:
        pad_h = step - pad_h
    if pad_w > 0:
        pad_w = step - pad_w

    # Pad the image with zeros (pad: left, right, top, bottom)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), "constant", 0)

    # Unfold the padded image into tiles
    tiles = image_padded.unfold(1, tile_size, step).unfold(2, tile_size, step)
    tiles = tiles.permute(1, 2, 0, 3, 4)

    # Output shape: (tile_H, tile_W, C, tile_size, tile_size)
    return tiles


def reconstruct_image_and_prediction(
    img_tiles: torch.Tensor,
    y_hat_tiles: torch.Tensor,
    original_size: tuple[int, int],
    overlap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstructs the image and predictions removing the overlap in tiles.
    Important is to pass prediction and image tiles in the correct shape!

    Args:
        img_tiles (torch.Tensor): Expects single tiled image of shape:
            (H_tile, W_tile, C, tile_size, tile_size)
        y_hat_tiles (torch.Tensor): Expects un-thresholded prediction of shape:
            (H_tile, W_tile, tile_size, tile_size, classes)
        original_size (tuple[int, int]): The size of the original image
        overlap (int): The overlap size of the tiles

    Returns:
        tuple(torch.Tensor, torch.Tensor): The stitched prediction and image of full size
    """
    H_tile, W_tile, C, tile_size, _ = img_tiles.size()

    stitch_image = torch.zeros(
        (C, H_tile * tile_size, W_tile * tile_size),
        device=img_tiles.device,
    )
    count_matrix = torch.zeros_like(stitch_image)

    stitch_pred = torch.zeros(
        (H_tile * tile_size, W_tile * tile_size, y_hat_tiles.size(4)),
        device=y_hat_tiles.device,
    )

    # step = 1024 - 125
    step = tile_size - overlap

    # Iterate over all the tiles
    for i in range(H_tile):
        for j in range(W_tile):
            # Calculate the start and end indices for the tiles
            start_h, start_w = i * step, j * step
            end_h, end_w = start_h + tile_size, start_w + tile_size

            # Add the tile to the stitched prediction
            stitch_pred[start_h:end_h, start_w:end_w] = torch.max(
                y_hat_tiles[i, j], stitch_pred[start_h:end_h, start_w:end_w]
            )

            # Add the tile to the stitched image and update count
            stitch_image[:, start_h:end_h, start_w:end_w] += img_tiles[i, j]
            count_matrix[:, start_h:end_h, start_w:end_w] += 1

    # Average the overlapping regions, avoid zero-division by clamping.
    stitch_image /= count_matrix.clamp(min=1)

    return (
        stitch_pred[: original_size[0], : original_size[1]],
        stitch_image[:, : original_size[0], : original_size[1]],
    )
