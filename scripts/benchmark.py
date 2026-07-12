"""Train a victim classifier and measure robustness to adversarial attacks.

Trains a small CNN, then sweeps the L-infinity budget epsilon and reports clean
and adversarial accuracy for FGSM, the basic iterative method, and the iterative
least-likely-class attack. Writes a robustness-vs-epsilon curve and a grid of
adversarial examples.

Usage:
    python scripts/benchmark.py --dataset mnist --epochs 3
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from advattack.attacks import bim, fgsm, iterative_least_likely
from advattack.data import image_loaders
from advattack.model import SmallCNN
from advattack.train import Trainer, accuracy, robust_accuracy, set_seed

EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--subset", type=int, default=6000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    train_loader, test_loader = image_loaders(args.dataset, subset=args.subset)
    model = Trainer(SmallCNN(), lr=1e-3).fit(train_loader, epochs=args.epochs).model

    clean = accuracy(model, test_loader)
    print(f"clean test accuracy: {clean:.4f}\n")

    curves: dict[str, list[float]] = {"fgsm": [], "bim": [], "ill": []}
    for eps in EPSILONS:
        acc_fgsm = robust_accuracy(model, test_loader, partial(fgsm, epsilon=eps))
        acc_bim = robust_accuracy(model, test_loader, partial(bim, epsilon=eps))
        acc_ill = robust_accuracy(
            model, test_loader, lambda m, x, y, e=eps: iterative_least_likely(m, x, e)
        )
        curves["fgsm"].append(acc_fgsm)
        curves["bim"].append(acc_bim)
        curves["ill"].append(acc_ill)
        print(f"eps={eps:.2f}  fgsm={acc_fgsm:.4f}  bim={acc_bim:.4f}  least_likely={acc_ill:.4f}")

    labels = {"fgsm": "FGSM", "bim": "BIM (iterative)", "ill": "iterative least-likely"}
    plt.figure(figsize=(7, 4.5))
    for key, curve in curves.items():
        plt.plot(EPSILONS, curve, marker="o", label=labels[key])
    plt.xlabel("epsilon (L-infinity budget)")
    plt.ylabel("accuracy under attack")
    plt.title(f"Adversarial robustness on {args.dataset}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_curve.png", dpi=120)
    plt.close()

    # Adversarial example grid at a fixed epsilon.
    x, y = next(iter(test_loader))
    x, y = x[:8], y[:8]
    x_adv = fgsm(model, x, y, epsilon=0.2)
    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        adv_pred = model(x_adv).argmax(dim=1)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
    for i in range(8):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].set_title(f"{int(clean_pred[i])}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(x_adv[i, 0], cmap="gray")
        axes[1, i].set_title(f"{int(adv_pred[i])}", fontsize=9)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("adv")
    fig.suptitle("Top: clean (prediction). Bottom: FGSM eps=0.2 (prediction).")
    plt.tight_layout()
    plt.savefig(out_dir / "adversarial_examples.png", dpi=120)
    plt.close()

    print(f"\nWrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
