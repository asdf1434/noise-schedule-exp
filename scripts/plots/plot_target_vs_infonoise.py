"""Compare rho*/w (Eq. 61 with the B.4 mmse, divided by the runs' loss weight w)
against the distribution the InfoNoise runs sampled from, pi (Eq. 15). Both are
densities per lambda = -2 log sigma.

    python scripts/plots/plot_target_vs_infonoise.py   # -> plots/pi_target_vs_infonoise.png
    python scripts/plots/plot_target_vs_infonoise.py --per_class
        # -> plots/pi_target_vs_infonoise_class{0..9}.png, rho* from each class's B.4 mmse

The target is restricted to the runs' sigma range [sigma_min, sigma_max] and
normalized there, as in Eq. (87). Run curves are the final refresh, median over
seeds, from results/infonoise_profiles.json.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.infonoise import loss_weight_of_sigma  # noqa: E402

DATASETS = ["mnist", "cifar10"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SCALES = ["x0.1", "", "x10"]


def per_lambda(sigma, density_per_log_sigma):
    """Density per log sigma -> density per lambda, normalized over the given points."""
    lam = -2 * np.log(sigma)
    d = density_per_log_sigma / 2
    return lam, d / np.trapezoid(d[::-1], lam[::-1])


def make_figure(runs, class_label=None):
    suffix = "" if class_label is None else f"_class{class_label}"
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for row, base in enumerate(DATASETS):
        for col, scale in enumerate(SCALES):
            name = f"{base}_{scale}" if scale else base
            run = runs[name]
            with open(f"results/mmse_b4/{name}{suffix}.json") as f:
                b4 = json.load(f)

            sigma = np.array(b4["sigma"])
            keep = (sigma >= run["sigma_min"]) & (sigma <= run["sigma_max"])
            sigma, mmse = sigma[keep], np.array(b4["mmse"])[keep]
            run_sigma = np.array(run["sigma"])

            w = loss_weight_of_sigma(sigma, "vpred", run["t_clip"])
            target = mmse / sigma**2 / w
            estimate = np.array(run["pi"])

            lam, target = per_lambda(sigma, target)
            run_lam, estimate = per_lambda(run_sigma, estimate)

            ax = axes[row, col]
            target_label = "ρ* / w (B.4 mmse)" if class_label is None else "ρ*_c / w (B.4 mmse, this class)"
            ax.plot(lam, target, color="#2a6fdb", linewidth=2, label=target_label)
            ax.plot(run_lam, estimate, color="#e8710a", linewidth=2, label="InfoNoise π (unconditional)" if class_label is not None else "InfoNoise π")
            inside = np.concatenate(
                [lam[target > 1e-3 * target.max()], run_lam[estimate > 1e-3 * estimate.max()]]
            )
            ax.set_xlim(inside.min() - 1, inside.max() + 1)
            title = f"{base} {scale or 'x1'}"
            if class_label is not None:
                title += f", digit {class_label}" if base == "mnist" else f", {CIFAR10_CLASSES[class_label]}"
            ax.set_title(title)
            ax.set_xlabel("λ = −2 log σ")
            ax.set_ylabel("density per λ")
            ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.text(
        0.5,
        -0.02,
        "w(σ) = 1/max(t_clip, 1−t)² = ((1+σ)/σ)² above the cap (t_clip = 0.05, or 0.005 for x0.1)",
        ha="center",
    )
    out = f"plots/pi_target_vs_infonoise{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


p = argparse.ArgumentParser()
p.add_argument("--per_class", action="store_true")
args = p.parse_args()

with open("results/infonoise_profiles.json") as f:
    runs = json.load(f)

if args.per_class:
    for c in range(10):
        make_figure(runs, c)
else:
    make_figure(runs)
