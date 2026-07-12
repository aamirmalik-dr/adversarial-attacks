import pytest
import torch
from torch.utils.data import DataLoader

from advattack.attacks import bim, fgsm, iterative_least_likely, pgd
from advattack.config import ATTACKS, SweepConfig, build_attack
from advattack.data import SyntheticImages, load_sample, sample_loader, save_sample
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
    assert torch.allclose(pgd(model, x, y, 0.0), x)


def test_pgd_within_epsilon_ball_and_range():
    model = SmallCNN().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.15
    x_adv = pgd(model, x, y, epsilon=eps, steps=7)
    assert torch.max(torch.abs(x_adv - x)) <= eps + 1e-6
    assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)


def test_pgd_random_start_is_stochastic():
    # Two runs from different seeds should differ because of the random start.
    model = SmallCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    set_seed(1)
    a = pgd(model, x, y, epsilon=0.2, steps=3)
    set_seed(2)
    b = pgd(model, x, y, epsilon=0.2, steps=3)
    assert not torch.allclose(a, b)


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


def test_sample_roundtrip(tmp_path):
    images = torch.rand(20, 1, 28, 28)
    labels = torch.randint(0, 10, (20,))
    path = tmp_path / "sample.npz"
    save_sample(images, labels, path)
    loaded_images, loaded_labels = load_sample(path)
    # uint8 quantization tolerance is 1/255.
    assert torch.max(torch.abs(loaded_images - images)) <= 1.0 / 255 + 1e-6
    assert torch.equal(loaded_labels, labels)
    loader = sample_loader(path, batch_size=8)
    assert sum(y.shape[0] for _, y in loader) == 20


def test_config_from_json_and_build_attack(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        '{"dataset": "mnist", "attacks": ["fgsm", "pgd"], "epsilons": [0.0, 0.1], '
        '"unknown_key": 42}',
        encoding="utf-8",
    )
    cfg = SweepConfig.from_json(cfg_path)
    assert cfg.attacks == ["fgsm", "pgd"]
    assert cfg.epsilons == [0.0, 0.1]
    model = SmallCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    for name in cfg.attacks:
        atk = build_attack(name, 0.1, steps=3)
        assert atk(model, x, y).shape == x.shape


def test_config_rejects_unknown_attack():
    with pytest.raises(ValueError):
        SweepConfig(attacks=["fgsm", "not_an_attack"])
    assert set(ATTACKS) == {"fgsm", "bim", "pgd", "least_likely"}
