# WRITTEN BY CLAUDE

"""Exp3 deliverable: FID (converged, averaged over a late-epoch window) as a
function of training sigma, for the logit_normal(mu=0, sigma) sweep -- one
line per sampling schedule. Complements plot.py / aggregate_fid.py's
FID-vs-epoch plots, which don't show this cross-sigma comparison directly.

Run from repo root after `python scripts/plots/aggregate_fid.py`:
    python -m scripts.plots.plot_sigma_sweep
"""

import argparse
import json
import re
import statistics

import matplotlib.pyplot as plt

from scripts.plots.plot_style import SCHEDULE_COLORS

SIGMA_RE = re.compile(r"^logit_normal_mu_0\.0_sigma_(?P<sigma>[\d.]+)$")


def _converged_mean_and_std(epoch_to_stats: dict, epoch_lo: int, epoch_hi: int):
    """Averages per-epoch across-seed means over [epoch_lo, epoch_hi], and
    propagates the across-seed stds (mean of variances) over the same window.
    Returns (mean, std) or (None, None) if no epochs fall in the window.
    """
    means, stds = [], []
    for epoch_key, stats in epoch_to_stats.items():
        numeric_epoch = "".join(c for c in epoch_key if c.isdigit())
        if not numeric_epoch:
            continue
        epoch = int(numeric_epoch)
        if epoch_lo <= epoch <= epoch_hi:
            means.append(stats["mean"])
            stds.append(stats["std"])
    if not means:
        return None, None
    mean = statistics.fmean(means)
    std = (statistics.fmean(s**2 for s in stds)) ** 0.5
    return mean, std


def plot_sigma_sweep(aggregated: dict, save_path: str, epoch_lo: int, epoch_hi: int):
    # sigma_data[schedule] = [(sigma, mean, std), ...]
    sigma_data: dict = {}

    for experiment_name, schedule_data in aggregated.items():
        match = SIGMA_RE.match(experiment_name)
        if match is None:
            continue
        sigma = float(match.group("sigma"))
        for schedule_name, epoch_to_stats in schedule_data.items():
            mean, std = _converged_mean_and_std(epoch_to_stats, epoch_lo, epoch_hi)
            if mean is None:
                continue
            sigma_data.setdefault(schedule_name, []).append((sigma, mean, std))

    if not sigma_data:
        print(
            "Error: no 'logit_normal_mu_0.0_sigma_<S>' base experiments found "
            "in aggregated results."
        )
        return

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    for schedule_name in sorted(sigma_data.keys()):
        points = sorted(sigma_data[schedule_name])
        sigmas = [p[0] for p in points]
        means = [p[1] for p in points]
        stds = [p[2] for p in points]
        color = SCHEDULE_COLORS.get(schedule_name)
        plt.errorbar(
            sigmas,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=3,
            label=schedule_name,
            color=color,
        )

    plt.title(
        f"Converged FID vs. Training Sigma (logit-normal, mu=0, "
        f"epochs {epoch_lo}-{epoch_hi} mean)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Training sigma (logit-normal, mu=0)", fontsize=13)
    plt.ylabel("FID (converged; lower is better)", fontsize=13)
    plt.legend(title="Sampling schedule", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot successfully saved to {save_path}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path", type=str, default="results/aggregated_fid_results.json"
    )
    parser.add_argument("--save_path", type=str, default="plots/exp3_sigma_sweep.png")
    parser.add_argument(
        "--epoch_window",
        type=int,
        nargs=2,
        default=[70, 100],
        metavar=("EPOCH_LO", "EPOCH_HI"),
        help="Inclusive epoch range to average over as the 'converged' FID (default: 70 100)",
    )
    args = parser.parse_args()

    with open(args.results_path) as f:
        aggregated = json.load(f)

    epoch_lo, epoch_hi = args.epoch_window
    plot_sigma_sweep(aggregated, args.save_path, epoch_lo, epoch_hi)


if __name__ == "__main__":
    main()
