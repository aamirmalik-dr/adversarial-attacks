"""Download MNIST or Fashion-MNIST through torchvision into the data directory.

Usage:
    python scripts/download_data.py --root data --dataset mnist
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    args = parser.parse_args()
    from torchvision import datasets

    cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    cls(root=args.root, train=True, download=True)
    cls(root=args.root, train=False, download=True)
    print(f"{args.dataset} downloaded to {args.root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
