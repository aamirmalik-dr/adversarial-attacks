"""Image data loading and a synthetic dataset for offline tests.

Images are kept in the raw [0, 1] pixel range, not normalized, so that
adversarial perturbations can be bounded and clipped directly in pixel space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

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


def save_sample(images: torch.Tensor, labels: torch.Tensor, path: str | Path) -> None:
    """Save an image sample to a compact ``.npz`` file.

    Pixels are stored as ``uint8`` in [0, 255] to keep the committed file small.

    Args:
        images: Float images in [0, 1] of shape ``(N, 1, 28, 28)``.
        labels: Integer labels of shape ``(N,)``.
        path: Destination ``.npz`` path.
    """
    imgs = (images.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    lbls = labels.detach().cpu().to(torch.int64).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=imgs, labels=lbls)


def load_sample(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a committed ``.npz`` image sample back into float tensors.

    Args:
        path: Path to a file written by :func:`save_sample`.

    Returns:
        A ``(images, labels)`` tuple with images in [0, 1] as float32.
    """
    data = np.load(Path(path))
    images = torch.from_numpy(data["images"].astype(np.float32) / 255.0)
    labels = torch.from_numpy(data["labels"].astype(np.int64))
    return images, labels


def sample_loader(path: str | Path, batch_size: int = 128) -> DataLoader:
    """Build a non-shuffling loader over a committed ``.npz`` sample."""
    images, labels = load_sample(path)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
