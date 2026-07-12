# Data

This directory is gitignored. No datasets are committed.

The victim classifier is trained on MNIST (or Fashion-MNIST), downloaded through
torchvision on first use:

```bash
python scripts/download_data.py --root data --dataset mnist
```

The unit tests use a small synthetic image dataset and need no download.
