"""Gradient-based adversarial attacks in the L-infinity norm.

All attacks operate on images in the [0, 1] pixel range and return perturbed
images clipped back to that range and to an epsilon-ball around the input. The
implementations are deliberately small and self-contained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fgsm(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Fast gradient sign method (Goodfellow et al., 2015).

    Takes a single step of size ``epsilon`` in the direction that increases the
    classification loss for the true labels ``y``.

    Args:
        model: The victim classifier (set to eval mode by the caller).
        x: Clean images of shape ``(batch, C, H, W)`` in [0, 1].
        y: True labels of shape ``(batch,)``.
        epsilon: L-infinity perturbation budget.

    Returns:
        Adversarial images of the same shape, clipped to [0, 1].
    """
    if epsilon == 0:
        return x.detach().clone()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def bim(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float | None = None,
    steps: int = 10,
) -> torch.Tensor:
    """Basic iterative method (Kurakin et al., 2017), an iterative FGSM.

    Applies ``steps`` small signed-gradient steps of size ``alpha`` and projects
    back into the L-infinity epsilon-ball after each step.

    Args:
        model: The victim classifier.
        x: Clean images in [0, 1].
        y: True labels.
        epsilon: L-infinity budget for the total perturbation.
        alpha: Per-step size. Defaults to ``epsilon / steps * 1.25``.
        steps: Number of iterations.

    Returns:
        Adversarial images clipped to [0, 1] and to the epsilon-ball.
    """
    if epsilon == 0:
        return x.detach().clone()
    if alpha is None:
        alpha = epsilon / steps * 1.25
    x_orig = x.detach()
    x_adv = x_orig.clone()
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0.0, 1.0)
    return x_adv


def iterative_least_likely(
    model: torch.nn.Module,
    x: torch.Tensor,
    epsilon: float,
    alpha: float | None = None,
    steps: int = 10,
) -> torch.Tensor:
    """Iterative least-likely-class attack (Kurakin et al., 2017).

    A targeted attack that iteratively pushes each input toward the class the
    model currently deems least likely, which tends to produce more damaging
    misclassifications than the untargeted method.

    Args:
        model: The victim classifier.
        x: Clean images in [0, 1].
        epsilon: L-infinity budget.
        alpha: Per-step size. Defaults to ``epsilon / steps * 1.25``.
        steps: Number of iterations.

    Returns:
        Adversarial images clipped to [0, 1] and to the epsilon-ball.
    """
    if epsilon == 0:
        return x.detach().clone()
    if alpha is None:
        alpha = epsilon / steps * 1.25
    x_orig = x.detach()
    with torch.no_grad():
        target = model(x_orig).argmin(dim=1)  # least likely class
    x_adv = x_orig.clone()
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), target)
        grad = torch.autograd.grad(loss, x_adv)[0]
        # Descend the loss toward the target class.
        x_adv = x_adv.detach() - alpha * grad.sign()
        delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0.0, 1.0)
    return x_adv
