# adversarial-attacks

Gradient-based adversarial attacks on image classifiers in PyTorch. Clean-room implementations of the fast gradient sign method (FGSM), the basic iterative method (BIM), and the iterative least-likely-class attack, with robustness-vs-epsilon curves that show how a well-trained classifier collapses under small, bounded perturbations.

## What it does

- Trains a small convolutional victim classifier on MNIST or Fashion-MNIST, kept in the raw [0, 1] pixel range so perturbations can be bounded directly in pixel space.
- Implements three L-infinity attacks from scratch: single-step FGSM, the iterative BIM (projected signed-gradient steps), and the targeted iterative least-likely-class attack.
- Sweeps the perturbation budget epsilon and reports accuracy under each attack, then writes a robustness curve and a grid of clean versus adversarial examples with predictions.
- Ships a synthetic-image test path so the unit tests and CI run with no download, checking that every attack stays inside its epsilon-ball and in [0, 1], and that attacks do not increase accuracy.

## What it does not do

- No adversarial training or certified defenses. This measures vulnerability; it does not harden the model.
- No L2 or L0 attacks, and no black-box or transfer attacks. All three methods are white-box and L-infinity.
- No large or high-resolution models. The victim is a compact CNN so the whole sweep runs on a CPU in minutes.

## Install

```
python -m venv .venv
.venv\Scripts\activate      # Windows, or: source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11 or newer. On Linux CI, install the CPU build of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Run

```
python scripts/download_data.py --root data --dataset mnist   # optional prefetch
python scripts/benchmark.py --dataset mnist --epochs 3         # train, attack, plot
pytest -q                                                      # tests, fully offline
```

The demo notebook is `notebooks/demo.ipynb`, executed with saved outputs.

## Results

Produced by `python scripts/benchmark.py --dataset mnist --epochs 3`: a small CNN trained on a 6000-image MNIST subset, evaluated on 2000 test images, single fixed seed. Clean test accuracy is 0.9630. The table gives accuracy under each attack as a function of the L-infinity budget epsilon (accuracy, so lower means a more successful attack).

| epsilon | FGSM | BIM (iterative) | iterative least-likely |
| --- | --- | --- | --- |
| 0.00 | 0.9630 | 0.9630 | 0.9630 |
| 0.05 | 0.9080 | 0.8940 | 0.9570 |
| 0.10 | 0.7930 | 0.7160 | 0.9390 |
| 0.15 | 0.5980 | 0.3895 | 0.8660 |
| 0.20 | 0.3780 | 0.1000 | 0.6240 |
| 0.30 | 0.0870 | 0.0000 | 0.0550 |

Three findings, all as observed. The single-step FGSM already halves accuracy by epsilon 0.15 and drives it near zero by 0.30. The iterative BIM is strictly stronger at every nonzero budget, reaching 0.10 accuracy at epsilon 0.20 and total failure (0.0000) at 0.30, because it takes several smaller gradient steps and re-projects, searching the epsilon-ball more thoroughly than one step can. The targeted least-likely-class attack degrades accuracy more slowly at small budgets, since forcing the specific least-likely label is harder than simply causing any error, but it too collapses the model once the budget is large. At these epsilons the perturbations are small in pixel space yet flip most predictions, the standard illustration of adversarial fragility.

## Package layout

```
src/advattack/      library code (model, data, attacks, trainer and metrics)
scripts/            download_data.py, benchmark.py
notebooks/          demo.ipynb with executed outputs
tests/              pytest suite, runs on synthetic data offline
data/               gitignored, MNIST downloaded on demand
```

## References

- Goodfellow, Shlens, Szegedy, Explaining and Harnessing Adversarial Examples, 2015 (FGSM).
- Kurakin, Goodfellow, Bengio, Adversarial Examples in the Physical World, 2017 (BIM and iterative least-likely).

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/dr-aamirmalik

## License

MIT, see LICENSE.
