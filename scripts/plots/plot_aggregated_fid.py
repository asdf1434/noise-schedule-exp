# WRITTEN BY CLAUDE

"""Shared plotting logic for a dataset's train-dist x schedule sweep, driven
off mean +/- std across seeds in results/aggregated_fid_results.json. Used by
plot_mnist.py and plot_eurosat.py so the two datasets render with an
identical color (train dist) / linestyle (schedule) scheme -- see
plot_style.py.

Produces (all under <plots_dir>/):
  <label>.png                    one curve per sampling schedule, per train dist
  schedule_<sched>.png           one curve per train dist, per fixed schedule
  fid_combined_all_<name>.png    every train dist x every schedule in one figure,
                                  color = train dist, linestyle = schedule
"""

import json
import os

import matplotlib.pyplot as plt

from scripts.plots.plot_style import SCHEDULE_COLORS, SCHEDULE_LINESTYLES, dist_color

AGGREGATED_PATH = "results/aggregated_fid_results.json"
SCHEDULES = ["uniform", "shifted_fine", "shifted_coarse", "logit_normal"]


def _epochs_means_stds(epoch_data: dict) -> tuple[list, list, list]:
    rows = []
    for epoch_key, stats in epoch_data.items():
        numeric = "".join(c for c in epoch_key if c.isdigit())
        if numeric:
            rows.append((int(numeric), stats["mean"], stats["std"]))
    rows.sort()
    if not rows:
        return [], [], []
    epochs, means, stds = zip(*rows)
    return list(epochs), list(means), list(stds)


def plot_per_dist(aggregated: dict, dist: str, dataset_label: str, save_path: str):
    if dist not in aggregated:
        print(f"  -> [SKIP] '{dist}' not in aggregated results")
        return
    plt.figure(figsize=(9, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    for schedule in SCHEDULES:
        epoch_data = aggregated[dist].get(schedule)
        if not epoch_data:
            continue
        epochs, means, stds = _epochs_means_stds(epoch_data)
        if not epochs:
            continue
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.plot(epochs, means, marker="o", linewidth=2, label=schedule, color=SCHEDULE_COLORS[schedule])
        plt.fill_between(epochs, lower, upper, alpha=0.15, color=SCHEDULE_COLORS[schedule])

    plt.title(f"{dataset_label} FID vs. Epoch by Sampling Schedule\n(Train Dist: {dist})", fontsize=14, fontweight="bold")
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("FID Score (mean ± std across seeds, lower is better)", fontsize=12)
    plt.legend(title="Sampling Schedule", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> saved {save_path}")


def plot_schedule_comparison(aggregated: dict, dists: list, schedule: str, dataset_label: str, save_path: str, prefix: str = ""):
    plt.figure(figsize=(9, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plotted_any = False
    for dist in dists:
        epoch_data = aggregated.get(dist, {}).get(schedule)
        if not epoch_data:
            continue
        epochs, means, stds = _epochs_means_stds(epoch_data)
        if not epochs:
            continue
        color = dist_color(dist)
        plt.plot(epochs, means, marker="o", linewidth=2, label=dist.removeprefix(prefix), color=color)
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.fill_between(epochs, lower, upper, alpha=0.1, color=color)
        plotted_any = True

    if not plotted_any:
        print(f"  -> [SKIP] no dists have data for schedule '{schedule}'")
        plt.close()
        return

    plt.title(f"{dataset_label} FID vs. Epoch by Training Distribution\n(Sampling Schedule: {schedule})", fontsize=14, fontweight="bold")
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("FID Score (mean ± std across seeds, lower is better)", fontsize=12)
    plt.legend(title="Training Distribution", fontsize=9, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> saved {save_path}")


def plot_combined_all(aggregated: dict, dists: list, dataset_label: str, save_path: str, prefix: str = ""):
    """One figure, every (train dist, schedule) pair as its own line: color
    encodes training distribution, linestyle encodes sampling schedule.
    """
    plt.figure(figsize=(18, 10))
    plt.style.use("seaborn-v0_8-whitegrid")

    for dist in dists:
        for schedule in SCHEDULES:
            epoch_data = aggregated.get(dist, {}).get(schedule)
            if not epoch_data:
                continue
            epochs, means, _ = _epochs_means_stds(epoch_data)
            if not epochs:
                continue
            plt.plot(
                epochs,
                means,
                marker=".",
                linewidth=2,
                color=dist_color(dist),
                linestyle=SCHEDULE_LINESTYLES[schedule],
            )

    dist_handles = [
        plt.Line2D([0], [0], color=dist_color(d), linewidth=2, label=d.removeprefix(prefix))
        for d in dists
    ]
    schedule_handles = [
        plt.Line2D([0], [0], color="black", linestyle=SCHEDULE_LINESTYLES[s], linewidth=2, label=s)
        for s in SCHEDULES
    ]
    dist_legend = plt.legend(
        handles=dist_handles, title="Training Distribution", loc="upper left",
        bbox_to_anchor=(1.01, 1.0), fontsize=10, handlelength=2.5,
    )
    plt.gca().add_artist(dist_legend)
    plt.legend(
        handles=schedule_handles, title="Sampling Schedule", loc="lower left",
        bbox_to_anchor=(1.01, 0.0), fontsize=10, handlelength=2.5,
    )

    plt.title(
        f"{dataset_label} FID vs. Training Epoch, All Training Distributions x Sampling Schedules (mean across seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("FID Score (Lower is Better)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(left=0)
    plt.subplots_adjust(right=0.78)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> saved {save_path}")


def run(dists: list, prefix: str, dataset_label: str, plots_dir: str, combined_name: str):
    """dists: full aggregated-results keys (e.g. "eurosat_uniform" or "uniform").
    prefix: stripped from each dist to build per-dist filenames/legend labels
    (e.g. "eurosat_" or "").
    """
    with open(AGGREGATED_PATH) as f:
        aggregated = json.load(f)

    os.makedirs(plots_dir, exist_ok=True)
    short = lambda d: d.removeprefix(prefix)

    print("Per-training-dist plots (one curve per sampling schedule):")
    for dist in dists:
        plot_per_dist(aggregated, dist, dataset_label, os.path.join(plots_dir, f"{short(dist)}.png"))

    print("\nPer-schedule comparison plots (one curve per training dist):")
    for schedule in SCHEDULES:
        plot_schedule_comparison(
            aggregated, dists, schedule, dataset_label,
            os.path.join(plots_dir, f"schedule_{schedule}.png"),
            prefix=prefix,
        )

    print("\nCombined master plot:")
    plot_combined_all(aggregated, dists, dataset_label, os.path.join(plots_dir, combined_name), prefix=prefix)
