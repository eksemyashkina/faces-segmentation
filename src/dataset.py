from typing import List, Tuple, Callable
from pathlib import Path
import PIL.Image
import numpy as np
import datasets
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        super().__init__()
        self.images_dir = Path(root) / "images" / subset
        self.masks_dir = Path(root) / "annotations" / subset
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted(list(Path(self.images_dir).glob("**/*.jpg")))
        self.masks = sorted(list(Path(self.masks_dir).glob("**/*.png")))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = PIL.Image.open(self.images[idx]).convert("RGB")
        mask = PIL.Image.open(self.masks[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


def collate_fn(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in items])
    masks = torch.stack([item[1] for item in items])
    return images, masks

