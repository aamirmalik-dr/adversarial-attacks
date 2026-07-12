"""Offline robustness sweep: the primary command-line entry point.

Loads a committed victim classifier and a committed image sample, sweeps the
L-infinity budget epsilon for each configured attack, and writes the robustness
curve (hero figure), an adversarial-example grid, ``metrics.json`` and
``RESULTS.md``. No training and no network access are required. Also installed
as the console command ``advattack-sweep``.

Usage:
    python scripts/sweep.py --config configs/mnist_sweep.json
"""

from __future__ import annotations

from advattack.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
