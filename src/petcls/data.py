"""Datasets, transforms and a leakage-free stratified validation split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import DataConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )


def eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation -> values in [0, 1] (CHW or NCHW)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


class _TransformSubset(Dataset):
    """Subset that applies its own transform (so train/val share files but not augmentation)."""

    def __init__(self, base: Dataset, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return self.transform(img), y


def _labels(ds) -> list[int]:
    if hasattr(ds, "_labels"):
        return list(ds._labels)
    return [int(ds[i][1]) for i in range(len(ds))]


def stratified_split(labels: list[int], val_fraction: float, seed: int):
    idx = np.arange(len(labels))
    if val_fraction <= 0:
        return idx, np.array([], dtype=int)
    tr, va = train_test_split(idx, test_size=val_fraction, stratify=labels, random_state=seed)
    return np.sort(tr), np.sort(va)


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list[str]


def build_datasets(cfg: DataConfig, seed: int = 42) -> DataBundle:
    size = cfg.img_size
    if cfg.dataset == "fake":
        n_cls = 37
        base = datasets.FakeData(cfg.fake_size, (3, size, size), n_cls, random_offset=0)
        test = datasets.FakeData(
            cfg.fake_size // 2,
            (3, size, size),
            n_cls,
            transform=eval_transform(size),
            random_offset=10_000,
        )
        labels = [int(base[i][1]) for i in range(len(base))]
        tr, va = (
            np.arange(len(labels))[: int(len(labels) * 0.8)],
            np.arange(len(labels))[int(len(labels) * 0.8) :],
        )
        names = [f"class_{i}" for i in range(n_cls)]
    elif cfg.dataset == "oxford_pets":
        base = datasets.OxfordIIITPet(
            cfg.root, split="trainval", target_types="category", download=True
        )
        test = datasets.OxfordIIITPet(
            cfg.root,
            split="test",
            target_types="category",
            download=True,
            transform=eval_transform(size),
        )
        labels = _labels(base)
        tr, va = stratified_split(labels, cfg.val_fraction, seed)
        names = list(base.classes)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}")

    return DataBundle(
        train=_TransformSubset(base, tr, train_transform(size)),
        val=_TransformSubset(base, va, eval_transform(size)),
        test=test,
        class_names=names,
    )


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


# Oxford-IIIT Pet convention: cat breeds are capitalised in the original file names.
CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx",
}


def species_of(class_name: str) -> str:
    return "cat" if class_name in CAT_BREEDS else "dog"


__all__ = [
    "DataBundle",
    "build_datasets",
    "make_loader",
    "train_transform",
    "eval_transform",
    "denormalize",
    "stratified_split",
    "species_of",
    "CAT_BREEDS",
]
