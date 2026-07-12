"""Training loop, clean accuracy, and adversarial (robust) accuracy."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Clean top-1 accuracy of ``model`` over ``loader``."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.shape[0]
    return correct / max(total, 1)


def robust_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    attack: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cpu",
) -> float:
    """Accuracy under an attack.

    ``attack`` is a callable taking ``(model, x, y)`` and returning perturbed
    images. Gradients are needed to craft the perturbation, so this is not run
    under ``torch.no_grad``; the accuracy count itself is done without grad.

    Args:
        model: The victim classifier.
        loader: Test data loader.
        attack: A function crafting adversarial examples from a clean batch.
        device: Torch device.

    Returns:
        Fraction of adversarial examples still classified correctly.
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.shape[0]
    return correct / max(total, 1)


@dataclass
class Trainer:
    """Trains the victim classifier with Adam and cross-entropy."""

    model: torch.nn.Module
    lr: float = 1e-3
    device: str = "cpu"

    def fit(self, train_loader: DataLoader, epochs: int = 3, verbose: bool = True) -> Trainer:
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            running, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * y.shape[0]
                n += y.shape[0]
            if verbose:
                print(f"epoch {epoch + 1:3d}  train_loss={running / max(n, 1):.4f}")
        return self
