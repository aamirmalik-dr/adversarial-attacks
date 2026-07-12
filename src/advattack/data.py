"""Image data loading and a synthetic dataset for offline tests.

Images are kept in the raw [0, 1] pixel range, not normalized, so that
adversarial perturbations can be bounded and clipped directly in pixel space.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

DATASETS = ("mnist", "fashion")


class SyntheticImages(Dataset):
    """Random 1x28x28 images in [0, 1] with labels tied to mean brightness.

    The label correlates with the mean pixel value, so a small classifier can
    reach above-chance accuracy. This lets attack tests assert that accuracy
    drops under perturbation without any download.
    """

    def __init__(self, n: int = 256, num_classes: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.images = rng.random((n, 1, 28, 28)).astype(np.float32)
        signal = self.images.mean(axis=(1, 2, 3))
        bins = np.quantile(signal, np.linspace(0, 1, num_classes + 1))
        labels = np.clip(np.digitize(signal, bins[1:-1]), 0, num_classes - 1)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), int(self.labels[idx])


def image_loaders(
    dataset: str = "mnist",
    root: str = "data",
    batch_size: int = 128,
    subset: int | None = 6000,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return train and test loaders for MNIST or Fashion-MNIST in [0, 1] pixels.

    Args:
        dataset: ``"mnist"`` or ``"fashion"``.
        root: Directory for the torchvision download (gitignored).
        batch_size: Batch size.
        subset: If set, use this many training and ``subset // 3`` test images.
        seed: Seed for the subsample.

    Returns:
        A ``(train_loader, test_loader)`` tuple.

    Raises:
        ValueError: If ``dataset`` is not recognized.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    from torchvision import datasets, transforms

    tf = transforms.ToTensor()  # scales to [0, 1], no normalization
    cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
    train = cls(root=root, train=True, download=True, transform=tf)
    test = cls(root=root, train=False, download=True, transform=tf)

    if subset is not None:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(len(train), size=min(subset, len(train)), replace=False)
        test_idx = rng.choice(len(test), size=min(subset // 3, len(test)), replace=False)
        train = Subset(train, train_idx.tolist())
        test = Subset(test, test_idx.tolist())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
