"""Plot results/mmse_b4/*.json: mmse against log-SNR lambda = -2 log sigma.

    python scripts/plots/plot_mmse_b4.py          # full grid -> plots/mmse_b4.png
    python scripts/plots/plot_mmse_b4.py --zoom   # -> plots/b4_zoomed.png

--zoom limits each panel's x-axis to where mmse is between 0.1% and 99.9% of
its maximum, plus 1 unit of lambda on each side.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--zoom", action="store_true")
args = p.parse_args()

DATASETS = ["mnist", "cifar10"]
SCALES = ["x0.1", "", "x10"]

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
for row, base in enumerate(DATASETS):
    for col, scale in enumerate(SCALES):
        name = f"{base}_{scale}" if scale else base
        with open(f"results/mmse_b4/{name}.json") as f:
            r = json.load(f)
        lam = -2 * np.log(np.array(r["sigma"]))
        ax = axes[row, col]
        mmse = np.array(r["mmse"])
        ax.plot(lam, mmse, color="#2a6fdb", linewidth=2)
        if args.zoom:
            inside = lam[(mmse > 1e-3 * mmse.max()) & (mmse < 0.999 * mmse.max())]
            ax.set_xlim(inside.min() - 1, inside.max() + 1)
        ax.set_title(f"{base} {scale or 'x1'}")
        ax.set_xlabel("λ = −2 log σ")
        ax.set_ylabel("mmse")
        ax.grid(alpha=0.3)

out = "plots/b4_zoomed.png" if args.zoom else "plots/mmse_b4.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
