# Results

Robustness sweep produced by `python scripts/sweep.py --config configs/mnist_sweep.json`, run fully offline from the committed victim weights and the committed 1000-image sample. Single fixed seed (0); iterative attacks use 10 steps.

Clean accuracy: **0.9590**

Accuracy under attack as a function of the L-infinity budget epsilon (lower means a more effective attack):

| epsilon | FGSM (single step) | BIM (iterative) | PGD (random start) | iterative least-likely |
| --- | --- | --- | --- | --- |
| 0.00 | 0.9590 | 0.9590 | 0.9590 | 0.9590 |
| 0.05 | 0.9050 | 0.8890 | 0.8880 | 0.9520 |
| 0.10 | 0.8020 | 0.7180 | 0.7090 | 0.9310 |
| 0.15 | 0.6070 | 0.4000 | 0.3740 | 0.8690 |
| 0.20 | 0.3880 | 0.1030 | 0.0800 | 0.6260 |
| 0.25 | 0.2140 | 0.0070 | 0.0040 | 0.2590 |
| 0.30 | 0.0900 | 0.0000 | 0.0000 | 0.0470 |

![Robustness curve](results/robustness_curve.png)

![Adversarial examples](results/adversarial_examples.png)
