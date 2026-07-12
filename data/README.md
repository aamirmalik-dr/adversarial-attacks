# Data

## Committed sample

`sample_mnist.npz` is a carved subset of the public MNIST test split: 1000
held-out test images with their labels, stored as uint8 pixels in [0, 255] to
keep the file small (about 160 KB). It is a real subset of MNIST, not synthetic.
The offline robustness sweep (`scripts/sweep.py`) reads it together with the
committed victim weights, so the whole experiment reproduces with no download.

Regenerate it (and the victim weights) with:

```bash
python scripts/train_victim.py --dataset mnist --epochs 3 --sample 1000
```

## Full dataset

The full MNIST and Fashion-MNIST datasets are gitignored and downloaded through
torchvision on demand, only needed to retrain the victim:

```bash
python scripts/download_data.py --root data --dataset mnist
```

MNIST is distributed by Yann LeCun and Corinna Cortes under the terms noted at
http://yann.lecun.com/exdb/mnist/. The unit tests use a small synthetic image
dataset and need no download.
