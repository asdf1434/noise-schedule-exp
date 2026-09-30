"""Plot results/q_target/*.json: the Eq. (13) target per lambda = -2 log sigma.

    python scripts/plots/plot_q_target.py   # -> plots/q_target.png

Each panel's x-axis covers where q is above 0.1% of its maximum, plus 1 unit
of lambda on each side.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["mnist", "cifar10"]
SCALES = ["x0.1", "", "x10"]

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
for row, base in enumerate(DATASETS):
    for col, scale in enumerate(SCALES):
        name = f"{base}_{scale}" if scale else base
        with open(f"results/q_target/{name}.json") as f:
            r = json.load(f)
        lam = np.array(r["lambda"])
        q = np.array(r["q_lambda"])
        ax = axes[row, col]
        ax.plot(lam, q, color="#2a6fdb", linewidth=2)
        inside = lam[q > 1e-3 * q.max()]
        ax.set_xlim(inside.min() - 1, inside.max() + 1)
        ax.set_title(f"{base} {scale or 'x1'}")
        ax.set_xlabel("λ = −2 log σ")
        ax.set_ylabel("q(λ)")
        ax.grid(alpha=0.3)

fig.savefig("plots/q_target.png", dpi=150)
print("wrote plots/q_target.png")
