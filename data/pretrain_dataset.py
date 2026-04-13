from pathlib import Path

from torch.utils.data import Dataset

from .common import (
    list_buffer_image_pairs,
    load_buffer_pair_image,
    apply_transform,
    to_image_tensor,
)


class PretrainDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform=None,
        return_filename: bool | None = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.return_filename = bool(return_filename)
        self.images = list_buffer_image_pairs(self.root)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        _, channel_0_path, channel_1_path = self.images[idx]
        image = load_buffer_pair_image(channel_0_path, channel_1_path)
        image, _ = apply_transform(self.transform, image, None)

        sample = {"image": to_image_tensor(image)}
        if self.return_filename:
            sample["name"] = channel_0_path.name
        return sample
