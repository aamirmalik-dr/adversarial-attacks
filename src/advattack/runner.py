"""Offline robustness-sweep runner shared by the CLI and the notebook.

Loads a committed victim classifier and a committed image sample, sweeps the
L-infinity budget epsilon for each configured attack, and writes the robustness
curve, an adversarial-example grid, ``metrics.json`` and ``RESULTS.md``. No
training and no network access are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from advattack.config import ATTACK_LABELS, SweepConfig, build_attack
from advattack.data import load_sample
from advattack.model import SmallCNN
from advattack.train import accuracy, robust_accuracy, set_seed


def load_victim(path: str) -> SmallCNN:
    """Load a saved victim ``state_dict`` into a fresh :class:`SmallCNN`."""
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run_sweep(cfg: SweepConfig) -> dict:
    """Run the epsilon sweep described by ``cfg`` and return a results dict."""
    set_seed(cfg.seed)
    model = load_victim(cfg.victim_path)
    images, labels = load_sample(cfg.sample_path)
    loader = DataLoader(TensorDataset(images, labels), batch_size=128, shuffle=False)

    clean = accuracy(model, loader)
    print(f"victim: {cfg.victim_path}")
    print(f"sample: {len(labels)} images from {cfg.sample_path}")
    print(f"clean accuracy: {clean:.4f}\n")

    curves: dict[str, list[float]] = {name: [] for name in cfg.attacks}
    for eps in cfg.epsilons:
        row = []
        for name in cfg.attacks:
            set_seed(cfg.seed)  # deterministic random start for PGD across the grid
            acc = robust_accuracy(model, loader, build_attack(name, eps, cfg.steps))
            curves[name].append(acc)
            row.append(f"{name}={acc:.4f}")
        print(f"eps={eps:.2f}  " + "  ".join(row))

    return {
        "dataset": cfg.dataset,
        "n_sample": int(len(labels)),
        "steps": cfg.steps,
        "seed": cfg.seed,
        "clean_accuracy": clean,
        "epsilons": cfg.epsilons,
        "attacks": cfg.attacks,
        "accuracy_under_attack": curves,
    }


def write_curve(cfg: SweepConfig, results: dict, out_dir: Path) -> Path:
    """Write the robustness-vs-epsilon hero figure."""
    plt.figure(figsize=(7, 4.5))
    for name in cfg.attacks:
        plt.plot(
            cfg.epsilons,
            results["accuracy_under_attack"][name],
            marker="o",
            label=ATTACK_LABELS.get(name, name),
        )
    plt.xlabel("epsilon (L-infinity budget)")
    plt.ylabel("accuracy under attack")
    plt.ylim(-0.02, 1.02)
    plt.title(f"Robustness curve on {cfg.dataset} (clean acc {results['clean_accuracy']:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "robustness_curve.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def write_grid(cfg: SweepConfig, out_dir: Path) -> Path:
    """Write a clean-versus-adversarial example grid at ``cfg.grid_epsilon``."""
    model = load_victim(cfg.victim_path)
    images, labels = load_sample(cfg.sample_path)
    n = 8
    x, y = images[:n], labels[:n]
    attack_name = "pgd" if "pgd" in cfg.attacks else cfg.attacks[-1]
    set_seed(cfg.seed)
    x_adv = build_attack(attack_name, cfg.grid_epsilon, cfg.steps)(model, x, y)
    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        adv_pred = model(x_adv).argmax(dim=1)

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        clean_ok = int(clean_pred[i]) == int(y[i])
        axes[0, i].set_title(
            f"{int(clean_pred[i])}", fontsize=10, color="black" if clean_ok else "red"
        )
        axes[0, i].axis("off")
        axes[1, i].imshow(x_adv[i, 0], cmap="gray", vmin=0, vmax=1)
        flipped = int(adv_pred[i]) != int(y[i])
        axes[1, i].set_title(
            f"{int(adv_pred[i])}", fontsize=10, color="red" if flipped else "black"
        )
        axes[1, i].axis("off")
    label = ATTACK_LABELS.get(attack_name, attack_name)
    fig.suptitle(
        f"Top: clean (prediction).  Bottom: {label} at epsilon={cfg.grid_epsilon} "
        "(prediction, red = flipped)."
    )
    plt.tight_layout()
    path = out_dir / "adversarial_examples.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def write_results_md(cfg: SweepConfig, results: dict, out_dir: Path) -> Path:
    """Render RESULTS.md at the repo root with the sweep table and figures."""
    lines = [
        "# Results",
        "",
        "Robustness sweep produced by `python scripts/sweep.py --config "
        f"configs/{cfg.dataset}_sweep.json`, run fully offline from the committed victim "
        f"weights and the committed {results['n_sample']}-image sample. Single fixed seed "
        f"({cfg.seed}); iterative attacks use {cfg.steps} steps.",
        "",
        f"Clean accuracy: **{results['clean_accuracy']:.4f}**",
        "",
        "Accuracy under attack as a function of the L-infinity budget epsilon (lower means a "
        "more effective attack):",
        "",
        "| epsilon | " + " | ".join(ATTACK_LABELS.get(a, a) for a in cfg.attacks) + " |",
        "| " + " | ".join(["---"] * (len(cfg.attacks) + 1)) + " |",
    ]
    for i, e in enumerate(cfg.epsilons):
        cells = [f"{results['accuracy_under_attack'][a][i]:.4f}" for a in cfg.attacks]
        lines.append(f"| {e:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![Robustness curve](results/robustness_curve.png)",
        "",
        "![Adversarial examples](results/adversarial_examples.png)",
        "",
    ]
    path = (out_dir / "..").resolve() / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the offline robustness sweep."""
    parser = argparse.ArgumentParser(description="Offline adversarial robustness sweep.")
    parser.add_argument("--config", default="configs/mnist_sweep.json")
    parser.add_argument("--out", default=None, help="override out_dir from the config")
    args = parser.parse_args(argv)

    cfg = SweepConfig.from_json(args.config)
    if args.out:
        cfg.out_dir = args.out
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_sweep(cfg)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    curve = write_curve(cfg, results, out_dir)
    grid = write_grid(cfg, out_dir)
    md = write_results_md(cfg, results, out_dir)

    print(f"\nwrote {out_dir / 'metrics.json'}")
    print(f"wrote {curve}")
    print(f"wrote {grid}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
