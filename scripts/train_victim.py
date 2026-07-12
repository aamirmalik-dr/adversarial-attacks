"""Train the victim classifier and carve a small committed image sample.

This is the one step that needs the full dataset. It trains the small CNN on a
subset of MNIST (or Fashion-MNIST), saves the weights to ``models/`` and carves
a few hundred test images to a compact ``.npz`` under ``data/``. With those two
artifacts committed, ``scripts/sweep.py`` runs the whole epsilon sweep offline
with no training and no download.

Usage:
    python scripts/train_victim.py --dataset mnist --epochs 3 --sample 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from advattack.data import image_loaders, save_sample
from advattack.model import SmallCNN
from advattack.train import Trainer, accuracy, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--subset", type=int, default=6000, help="training images")
    parser.add_argument("--sample", type=int, default=1000, help="test images to carve")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", default="models/victim_mnist.pt")
    parser.add_argument("--sample-out", default="data/sample_mnist.npz")
    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, test_loader = image_loaders(args.dataset, subset=args.subset, seed=args.seed)
    model = Trainer(SmallCNN(), lr=1e-3).fit(train_loader, epochs=args.epochs).model

    clean = accuracy(model, test_loader)
    print(f"clean test accuracy (train-time subset): {clean:.4f}")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"saved victim weights to {model_path}")

    # Carve the committed sample from the held-out test images.
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    images = torch.cat(xs)[: args.sample]
    labels = torch.cat(ys)[: args.sample]
    save_sample(images, labels, args.sample_out)
    print(f"carved {len(labels)} test images to {args.sample_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
