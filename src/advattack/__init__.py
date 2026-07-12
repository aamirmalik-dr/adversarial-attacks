"""Gradient-based adversarial attacks on image classifiers.

A small convolutional victim model, clean-room implementations of the fast
gradient sign method and its iterative variants (basic iterative method and the
iterative least-likely-class attack), and utilities to measure accuracy as a
function of the perturbation budget epsilon.
"""

from advattack.attacks import bim, fgsm, iterative_least_likely
from advattack.data import SyntheticImages, image_loaders
from advattack.model import SmallCNN
from advattack.train import Trainer, accuracy, robust_accuracy, set_seed

__all__ = [
    "fgsm",
    "bim",
    "iterative_least_likely",
    "SyntheticImages",
    "image_loaders",
    "SmallCNN",
    "Trainer",
    "accuracy",
    "robust_accuracy",
    "set_seed",
]

__version__ = "0.1.0"
