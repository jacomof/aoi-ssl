from __future__ import annotations

from pathlib import Path
from typing import Callable
import re

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BUFFER_STEM_PATTERN = re.compile(r"^(?P<base>.+)_buffer_?(?P<idx>[01])$")


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(
        [
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def split_buffer_stem(stem: str) -> tuple[str, int] | None:
    match = BUFFER_STEM_PATTERN.match(stem)
    if not match:
        return None
    return match.group("base"), int(match.group("idx"))


def list_buffer_image_pairs(
    path: Path,
    excluded_parts: set[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    if not path.exists():
        return []

    excluded_parts = {part.lower() for part in (excluded_parts or set())}
    candidates = list_images(path)
    grouped: dict[tuple[Path, str], dict[int, Path]] = {}

    for candidate in candidates:
        if excluded_parts and any(
            part.lower() in excluded_parts for part in candidate.parts
        ):
            continue

        stem_info = split_buffer_stem(candidate.stem)
        if stem_info is None:
            continue

        base_stem, idx = stem_info
        key = (candidate.parent, base_stem)
        grouped.setdefault(key, {})[idx] = candidate

    pairs = []
    for (parent, base_stem), channels in grouped.items():
        if 0 in channels and 1 in channels:
            pairs.append((base_stem, channels[0], channels[1]))

    pairs.sort(key=lambda item: (item[1].parent.as_posix(), item[0]))
    return pairs


def ensure_channels(image: np.ndarray, channels: int = 2) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == channels:
        return image
    if image.shape[-1] > channels:
        return image[..., :channels]
    repeats = channels - image.shape[-1]
    tail = np.repeat(image[..., -1:], repeats, axis=-1)
    return np.concatenate([image, tail], axis=-1)


def load_image(path: Path, channels: int = 2) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 3 and image.shape[-1] >= 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ensure_channels(image, channels=channels)


def load_buffer_pair_image(channel_0_path: Path, channel_1_path: Path) -> np.ndarray:
    channel_0 = cv2.imread(str(channel_0_path), cv2.IMREAD_UNCHANGED)
    channel_1 = cv2.imread(str(channel_1_path), cv2.IMREAD_UNCHANGED)

    if channel_0 is None:
        raise FileNotFoundError(f"Could not read image: {channel_0_path}")
    if channel_1 is None:
        raise FileNotFoundError(f"Could not read image: {channel_1_path}")

    channel_0 = channel_0[..., 0] if channel_0.ndim == 3 else channel_0
    channel_1 = channel_1[..., 0] if channel_1.ndim == 3 else channel_1

    if channel_0.shape[:2] != channel_1.shape[:2]:
        raise ValueError(
            "Buffer pair images must have the same spatial dimensions: "
            f"{channel_0_path} vs {channel_1_path}"
        )

    return np.stack([channel_0, channel_1], axis=-1)


def _one_hot_from_label_map(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((*label_map.shape, num_classes), dtype=np.float32)
    for cls_idx in range(num_classes):
        one_hot[..., cls_idx] = (label_map == cls_idx).astype(np.float32)
    return one_hot


def find_label_for_image(image_path: Path) -> Path | None:
    parent = image_path.parent
    stem = image_path.stem

    candidates = []
    # Only search explicit label locations to avoid picking deployment/evaluable masks.
    for lbl_dir_name in ("lbl", "label", "labels"):
        candidates.extend(
            [
                parent / lbl_dir_name / f"{stem}.png",
                parent / lbl_dir_name / f"{stem}.npy",
            ]
        )

    parts = list(image_path.parts)
    for src, dst in (("img", "lbl"), ("images", "labels")):
        if src in parts:
            idx = parts.index(src)
            replaced = Path(*parts[:idx], dst, *parts[idx + 1 :])
            candidates.extend(
                [
                    replaced.with_suffix(".png"),
                    replaced.with_suffix(".npy"),
                ]
            )

    candidates.extend(
        [
            parent / f"{stem}_label.png",
            parent / f"{stem}_lbl.png",
            parent / f"{stem}_label.npy",
            parent / f"{stem}_lbl.npy",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mask(
    mask_path: Path | None, height: int, width: int, num_classes: int
) -> np.ndarray:
    if mask_path is None:
        return np.zeros((height, width, num_classes), dtype=np.float32)

    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        return np.zeros((height, width, num_classes), dtype=np.float32)

    if mask.ndim == 3 and mask.shape[-1] == num_classes:
        return mask.astype(np.float32)

    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if mask.max() <= 1 and num_classes > 1:
        class_map = (mask > 0).astype(np.int32)
        return _one_hot_from_label_map(class_map, num_classes)

    return _one_hot_from_label_map(mask.astype(np.int32), num_classes)


def apply_transform(
    transform: Callable | None,
    image: np.ndarray,
    class_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | torch.Tensor | list, np.ndarray | torch.Tensor | None]:
    if transform is None:
        return image, class_mask

    try:
        transformed = transform(image=image, mask=class_mask)
    except TypeError:
        transformed = transform(image=image)

    if isinstance(transformed, dict):
        out_image = transformed.get("image", image)
        out_mask = transformed.get("mask", class_mask)
        return out_image, out_mask

    return transformed, class_mask


def to_image_tensor(image: np.ndarray | torch.Tensor | list) -> torch.Tensor | list:
    if isinstance(image, list):
        return [to_image_tensor(item) for item in image]
    if isinstance(image, torch.Tensor):
        return image

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    image = np.moveaxis(image, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(image))


def to_class_mask_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask.float()
    mask = np.asarray(mask, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(mask))


def infer_manufacturer_and_device(path: Path) -> tuple[str, str]:
    parts = path.parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return "unknown", parts[-2]
    return "unknown", "unknown"
