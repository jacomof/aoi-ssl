from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .common import (
    list_buffer_image_pairs,
    split_buffer_stem,
    load_buffer_pair_image,
    find_label_for_image,
    load_mask,
    apply_transform,
    to_image_tensor,
    to_class_mask_tensor,
    infer_manufacturer_and_device,
)


class RetrievalDataset(Dataset):
    def __init__(
        self,
        images: list[str | Path] | str | Path,
        classes: list[str],
        transform=None,
        return_filename: bool | None = False,
        return_manufacturer: bool | None = False,
        return_device: bool | None = False,
    ):
        if isinstance(images, (str, Path)):
            image_root = Path(images)
            img_dir = image_root / "img"
            image_root = img_dir if img_dir.exists() else image_root
            self.images = list_buffer_image_pairs(
                image_root,
                excluded_parts={"lbl", "label", "labels", "mask", "masks"},
            )
        else:
            path_images = [Path(p) for p in images]
            grouped: dict[tuple[Path, str], dict[int, Path]] = {}
            for image_path in path_images:
                stem_info = split_buffer_stem(image_path.stem)
                if stem_info is None:
                    continue
                base_stem, idx = stem_info
                key = (image_path.parent, base_stem)
                grouped.setdefault(key, {})[idx] = image_path

            pairs = []
            for (parent, base_stem), channels in grouped.items():
                if 0 in channels and 1 in channels:
                    pairs.append((base_stem, channels[0], channels[1]))
            pairs.sort(key=lambda item: (item[1].parent.as_posix(), item[0]))
            self.images = pairs

        self.classes = classes
        self.transform = transform
        self.return_filename = bool(return_filename)
        self.return_manufacturer = bool(return_manufacturer)
        self.return_device = bool(return_device)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        sample_stem, channel_0_path, channel_1_path = self.images[idx]
        image = load_buffer_pair_image(channel_0_path, channel_1_path)
        h, w = image.shape[:2]

        base_image_path = channel_0_path.with_name(
            f"{sample_stem}{channel_0_path.suffix}"
        )
        mask_path = find_label_for_image(base_image_path) or find_label_for_image(
            channel_0_path
        )
        class_mask = load_mask(mask_path, h, w, num_classes=len(self.classes))

        image, class_mask = apply_transform(self.transform, image, class_mask)

        class_mask_tensor = to_class_mask_tensor(class_mask)

        sample = {
            "image": to_image_tensor(image),
            "class_mask": class_mask_tensor,
            # Uncomment if ignore masks are available and should be used in training
            # "ignore_mask": torch.zeros_like(class_mask_tensor),
        }

        if self.return_filename:
            sample["name"] = channel_0_path.name

        manufacturer, device = infer_manufacturer_and_device(channel_0_path)
        if self.return_manufacturer:
            sample["manufacturer"] = manufacturer
        if self.return_device:
            sample["device"] = device

        return sample
