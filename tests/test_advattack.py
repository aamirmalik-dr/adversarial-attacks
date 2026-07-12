import torch
from torch.utils.data import DataLoader

from advattack.attacks import bim, fgsm, iterative_least_likely
from advattack.data import SyntheticImages
from advattack.model import SmallCNN
from advattack.train import Trainer, accuracy, robust_accuracy, set_seed


def _trained_model():
    set_seed(0)
    train = DataLoader(SyntheticImages(n=256, seed=0), batch_size=64, shuffle=True)
    return Trainer(SmallCNN(), lr=1e-3).fit(train, epochs=3, verbose=False).model


def test_model_output_shape():
    x = torch.rand(4, 1, 28, 28)
    assert SmallCNN()(x).shape == (4, 10)


def test_fgsm_respects_epsilon_and_range():
    model = SmallCNN().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.1
    x_adv = fgsm(model, x, y, epsilon=eps)
    assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)
    assert torch.max(torch.abs(x_adv - x)) <= eps + 1e-6


def test_zero_epsilon_is_identity():
    model = SmallCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    assert torch.allclose(fgsm(model, x, y, 0.0), x)
    assert torch.allclose(bim(model, x, y, 0.0), x)


def test_bim_within_epsilon_ball():
    model = SmallCNN().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.15
    x_adv = bim(model, x, y, epsilon=eps, steps=5)
    assert torch.max(torch.abs(x_adv - x)) <= eps + 1e-6
    assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)


def test_least_likely_within_epsilon_ball():
    model = SmallCNN().eval()
    x = torch.rand(6, 1, 28, 28)
    eps = 0.2
    x_adv = iterative_least_likely(model, x, epsilon=eps, steps=5)
    assert torch.max(torch.abs(x_adv - x)) <= eps + 1e-6


def test_attack_reduces_accuracy():
    model = _trained_model()
    loader = DataLoader(SyntheticImages(n=256, seed=1), batch_size=64)
    clean = accuracy(model, loader)
    from functools import partial

    adv = robust_accuracy(model, loader, partial(fgsm, epsilon=0.3))
    assert adv <= clean  # an attack should not improve accuracy
