"""JSON-backed configuration for epsilon-sweep experiments.

A sweep is fully described by a small JSON file: which victim weights and image
sample to load, which attacks to run, the epsilon grid, and per-attack iteration
counts. Keeping the configuration in data (not code) lets the same runner drive
different experiments and keeps runs reproducible from a committed file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch

from advattack.attacks import bim, fgsm, iterative_least_likely, pgd

ATTACKS = ("fgsm", "bim", "pgd", "least_likely")

# Human-readable labels used in figures and tables, keyed by attack name.
ATTACK_LABELS = {
    "fgsm": "FGSM (single step)",
    "bim": "BIM (iterative)",
    "pgd": "PGD (random start)",
    "least_likely": "iterative least-likely",
}


@dataclass
class SweepConfig:
    """Description of one robustness sweep.

    Attributes:
        dataset: Source dataset name, recorded for provenance only.
        victim_path: Path to the saved victim ``state_dict``.
        sample_path: Path to the committed ``.npz`` image sample.
        epsilons: L-infinity budgets to evaluate, starting at 0.0 for clean.
        attacks: Attack names to run, a subset of :data:`ATTACKS`.
        steps: Iteration count for the iterative attacks.
        grid_epsilon: Epsilon used for the adversarial-example grid figure.
        seed: Global seed, fixed for reproducibility.
        out_dir: Directory for figures, metrics, and the results table.
    """

    dataset: str = "mnist"
    victim_path: str = "models/victim_mnist.pt"
    sample_path: str = "data/sample_mnist.npz"
    epsilons: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    attacks: list[str] = field(default_factory=lambda: list(ATTACKS))
    steps: int = 10
    grid_epsilon: float = 0.2
    seed: int = 0
    out_dir: str = "results"

    def __post_init__(self) -> None:
        unknown = [a for a in self.attacks if a not in ATTACKS]
        if unknown:
            raise ValueError(f"unknown attack(s) {unknown}; choose from {ATTACKS}")

    @classmethod
    def from_json(cls, path: str | Path) -> SweepConfig:
        """Load a config from a JSON file, ignoring unknown keys."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def build_attack(
    name: str, epsilon: float, steps: int
) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a ``(model, x, y) -> x_adv`` callable for the named attack.

    Args:
        name: One of :data:`ATTACKS`.
        epsilon: L-infinity budget bound into the returned callable.
        steps: Iteration count for iterative attacks.

    Returns:
        A callable that crafts adversarial images for a clean batch.

    Raises:
        ValueError: If ``name`` is not a known attack.
    """
    if name == "fgsm":
        return partial(fgsm, epsilon=epsilon)
    if name == "bim":
        return partial(bim, epsilon=epsilon, steps=steps)
    if name == "pgd":
        return partial(pgd, epsilon=epsilon, steps=steps)
    if name == "least_likely":
        return lambda m, x, y: iterative_least_likely(m, x, epsilon, steps=steps)
    raise ValueError(f"unknown attack {name!r}; choose from {ATTACKS}")
