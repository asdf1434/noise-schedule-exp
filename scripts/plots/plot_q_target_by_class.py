"""Plot the per-class targets q(lambda) against the unconditional target, one
panel per cell, from results/q_target/.

    python scripts/plots/plot_q_target_by_class.py   # -> plots/q_target_by_class.png

Each panel's x-axis covers where any curve is above 0.1% of its maximum, plus
1 unit of lambda on each side.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["mnist", "cifar10"]
SCALES = ["x0.1", "", "x10"]


def load(name):
    with open(f"results/q_target/{name}.json") as f:
        r = json.load(f)
    return np.array(r["lambda"]), np.array(r["q_lambda"])


fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
for row, base in enumerate(DATASETS):
    for col, scale in enumerate(SCALES):
        name = f"{base}_{scale}" if scale else base
        ax = axes[row, col]
        visible = []
        for c in range(10):
            lam, q = load(f"{name}_class{c}")
            ax.plot(lam, q, color="#9aa7bd", linewidth=1, label="per class (10)" if c == 0 else None)
            visible.append(lam[q > 1e-3 * q.max()])
        lam, q = load(name)
        ax.plot(lam, q, color="#2a6fdb", linewidth=2.5, label="unconditional")
        visible.append(lam[q > 1e-3 * q.max()])
        visible = np.concatenate(visible)
        ax.set_xlim(visible.min() - 1, visible.max() + 1)
        ax.set_title(f"{base} {scale or 'x1'}")
        ax.set_xlabel("λ = −2 log σ")
        ax.set_ylabel("q(λ)")
        ax.grid(alpha=0.3)

axes[0, 0].legend(loc="upper left", fontsize=8)
fig.savefig("plots/q_target_by_class.png", dpi=150)
print("wrote plots/q_target_by_class.png")
