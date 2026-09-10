# WRITTEN BY CLAUDE

"""The noise distribution InfoNoise's auto-gate arm actually trains on.

Two arms only -- the fixed logit-normal(0,1) prior a practitioner would use, and
InfoNoise run as published (automatic gate, nothing about the data scale
supplied). Everything else that appears in plot_infonoise_allocation.py (the
hand-set gate, the oracle logit-normal) is deliberately absent: those arms
answer a different question.

The plotted quantity is `pi`, the density training samples are actually drawn
from, read from logs/metrics/<exp>/infonoise_profile.jsonl. `pi = rho_hat / w`
with the loss weight w fixed in sigma, so pi is what the model saw; rho_hat is
the target profile and is drawn faintly behind it for reference.

Left column: the final refresh, mean +/- s.e.m. over seeds.
Right column: how pi moved over training, one line per refresh, so the warm-up
cost visible in the FID curves has something to be read against.

    python scripts/plots/plot_auto_training_dist.py
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# dataset -> (data scale k, the auto arm's dist name)
ARMS = {
    "mnist": (1.0, "infonoise"),
    "mnist_x10": (10.0, "infonoise_sigma_max_400.0"),
    "mnist_x0.1": (0.1, "infonoise_sigma_min_0.0002"),
}
COLORS = {"mnist": "#4C72B0", "mnist_x10": "#DD8452", "mnist_x0.1": "#55A868"}


def load_runs(dataset: str):
    """-> list of {refresh: row} dicts, one per seed."""
    _, dist = ARMS[dataset]
    runs = []
    for path in sorted(glob.glob(
        f"logs/metrics/ds-{dataset}__cond-none__dist-{dist}__seed-*/infonoise_profile.jsonl"
    )):
        by_refresh = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                by_refresh[row["refresh"]] = row
        if by_refresh:
            runs.append(by_refresh)
    return runs


def _norm(a):
    """Unit mass. The grid is log-spaced, so summing over bins suffices."""
    a = np.asarray(a, dtype=float)
    return a / a.sum()


def _median(sigma, dens):
    return sigma[np.searchsorted(np.cumsum(dens) / dens.sum(), 0.5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="plots/infonoise_scaling")
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # The prior is logit-normal(0,1) on t, and logit(t) = -log(sigma), so on the
    # log-sigma axis it is exactly a standard normal centred on sigma=1.
    grid = np.logspace(-4.5, 3, 700)
    prior = _norm(np.exp(-0.5 * np.log(grid) ** 2))

    for row, (ds, (k, _dist)) in enumerate(ARMS.items()):
        runs = load_runs(ds)
        if not runs:
            print(f"[warn] no profile logs for {ds}")
            continue
        color = COLORS[ds]
        sigma = np.array(runs[0][max(runs[0])]["sigma_centers"])

        # --- left: final refresh, across seeds ---
        ax = axes[row][0]
        pi = np.array([_norm(r[max(r)]["pi"]) for r in runs])
        rho = np.array([_norm(r[max(r)]["rho_hat"]) for r in runs])
        sem = pi.std(axis=0, ddof=1) / np.sqrt(len(runs))
        ax.plot(grid, prior, color="0.45", ls="--", lw=2.0,
                label=f"logit-normal(0,1) prior   median sigma {_median(grid, prior):.3g}")
        ax.plot(sigma, rho.mean(axis=0), color=color, lw=1.2, alpha=0.45,
                label=r"InfoNoise $\hat{\rho}$ (target profile, for reference)")
        ax.plot(sigma, pi.mean(axis=0), color=color, lw=2.4,
                label=f"InfoNoise auto, trained on   median sigma "
                      f"{_median(sigma, pi.mean(axis=0)):.3g}   (n={len(runs)})")
        ax.fill_between(sigma, pi.mean(axis=0) - sem, pi.mean(axis=0) + sem,
                        color=color, alpha=0.2, lw=0)
        ax.set_title(f"{ds}  (k={k:g})   final refresh", fontsize=12, fontweight="bold")
        ax.set_ylabel("density per unit $\\log\\sigma$")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.92)

        # --- right: evolution over refreshes, seed 0 ---
        ax = axes[row][1]
        r0 = runs[0]
        refreshes = sorted(r0)
        cmap = plt.cm.viridis(np.linspace(0, 1, len(refreshes)))
        for c, ref in zip(cmap, refreshes):
            ax.plot(sigma, _norm(r0[ref]["pi"]), color=c, lw=1.0, alpha=0.85)
        ax.plot(grid, prior, color="0.45", ls="--", lw=1.8, label="prior (start point)")
        ax.set_title(f"{ds}   how it moved during training (seed 0, "
                     f"{len(refreshes)} refreshes, dark->light)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")

        for ax in axes[row]:
            ax.axvline(k, color=color, ls=":", lw=1.6)
            ax.text(k, ax.get_ylim()[1] * 0.96, f"  sigma={k:g}\n  (where the\n  signal is)",
                    fontsize=8, color=color, va="top")
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1e3)
            ax.grid(alpha=0.25, which="both")

    for ax in axes[-1]:
        ax.set_xlabel(r"noise level  $\sigma = (1-t)/t$")
    fig.suptitle("What noise levels InfoNoise (auto gate) actually trained on, "
                 "vs the logit-normal(0,1) prior", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "auto_training_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
