# WRITTEN BY CLAUDE

import argparse
import json
import re
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Matches e.g. "uniform_seed0" -> ("uniform", "0"),
# "logit_normal_mu_0.0_sigma_1.0_seed3" -> ("logit_normal_mu_0.0_sigma_1.0", "3")
SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def split_base_and_seed(experiment_name: str) -> tuple[str, str | None]:
    match = SEED_SUFFIX_RE.match(experiment_name)
    if match is None:
        return experiment_name, None
    return match.group("base"), match.group("seed")


def aggregate(master_data: dict) -> dict:
    """Groups per-seed experiments by their base train_dist name and computes
    mean/std FID across seeds for every (schedule, epoch) cell.

    Returns {base_name: {schedule_name: {epoch_key: {"mean", "std", "n", "seeds"}}}}
    """
    # grouped[base][schedule][epoch] = [(seed, score), ...]
    grouped: dict = {}

    for experiment_name, schedule_data in master_data.items():
        base_name, seed = split_base_and_seed(experiment_name)
        if seed is None:
            print(
                f"  -> [SKIP] '{experiment_name}' has no _seedN suffix, "
                "can't be aggregated across seeds"
            )
            continue
        for schedule_name, epoch_to_score in schedule_data.items():
            for epoch_key, score in epoch_to_score.items():
                grouped.setdefault(base_name, {}).setdefault(
                    schedule_name, {}
                ).setdefault(epoch_key, []).append((seed, score))

    aggregated = {}
    for base_name, schedule_data in grouped.items():
        for schedule_name, epoch_data in schedule_data.items():
            for epoch_key, seed_scores in epoch_data.items():
                scores = [score for _, score in seed_scores]
                mean = statistics.fmean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                aggregated.setdefault(base_name, {}).setdefault(schedule_name, {})[
                    epoch_key
                ] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "n": len(scores),
                    "seeds": sorted(seed for seed, _ in seed_scores),
                }

    return aggregated


def _epochs_and_stats(epoch_to_stats: dict) -> tuple[list, list, list]:
    epochs, means, stds = [], [], []
    for epoch_key, stats in epoch_to_stats.items():
        numeric_epoch = "".join(c for c in epoch_key if c.isdigit())
        if numeric_epoch:
            epochs.append(int(numeric_epoch))
            means.append(stats["mean"])
            stds.append(stats["std"])
    sorted_triples = sorted(zip(epochs, means, stds))
    if not sorted_triples:
        return [], [], []
    sorted_epochs, sorted_means, sorted_stds = zip(*sorted_triples)
    return list(sorted_epochs), list(sorted_means), list(sorted_stds)


def plot_experiment(aggregated: dict, experiment: str, save_path: str):
    """One figure, one curve (mean +/- std across seeds) per sampling schedule,
    for a single training-dist base experiment."""
    if experiment not in aggregated:
        print(f"Error: Experiment '{experiment}' not found in aggregated results.")
        print(f"Available base experiments: {list(aggregated.keys())}")
        return

    schedule_data = aggregated[experiment]

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plotted_any = False
    for schedule_name in sorted(schedule_data.keys()):
        epochs, means, stds = _epochs_and_stats(schedule_data[schedule_name])
        if not epochs:
            continue
        plt.plot(epochs, means, marker="o", linewidth=2, label=schedule_name)
        plt.fill_between(
            epochs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
        )
        plotted_any = True

    if not plotted_any:
        print(f"Error: No numeric epoch milestones found for '{experiment}'.")
        plt.close()
        return

    plt.title(
        f"FID vs. Training Epoch by Sampling Schedule (Train Dist: {experiment}, "
        "mean +/- std across seeds)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Training Epoch", fontsize=13)
    plt.ylabel("FID Score (Lower is Better)", fontsize=13)
    plt.legend(title="Sampling Schedule", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot successfully saved to {save_path} for experiment '{experiment}'!")


def plot_schedule_comparison(aggregated: dict, schedule: str, save_path: str):
    """One figure, one curve (mean +/- std across seeds) per training-dist base
    experiment, for a single fixed sampling schedule."""
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plotted_any = False
    for experiment in sorted(aggregated.keys()):
        epoch_to_stats = aggregated[experiment].get(schedule)
        if not epoch_to_stats:
            continue
        epochs, means, stds = _epochs_and_stats(epoch_to_stats)
        if not epochs:
            continue
        plt.plot(epochs, means, marker="o", linewidth=2, label=experiment)
        plt.fill_between(
            epochs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
        )
        plotted_any = True

    if not plotted_any:
        print(f"Error: No experiments have data for sampling schedule '{schedule}'.")
        plt.close()
        return

    plt.title(
        f"FID vs. Training Epoch by Training Distribution (Sampling: {schedule}, "
        "mean +/- std across seeds)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Training Epoch", fontsize=13)
    plt.ylabel("FID Score (Lower is Better)", fontsize=13)
    plt.legend(title="Training Distribution", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot successfully saved to {save_path} for sampling schedule '{schedule}'!")


def plot_all_combined(aggregated: dict, save_path: str):
    """One figure with every (training distribution, sampling schedule) curve
    plotted together: color encodes training distribution, line style encodes
    sampling schedule. No error bands -- with this many overlapping curves,
    mean +/- std shading is unreadable, so this shows means only.
    """
    experiments = sorted(aggregated.keys())
    schedules = sorted({s for exp in experiments for s in aggregated[exp].keys()})

    color_cycle = plt.cm.tab10.colors
    exp_colors = {exp: color_cycle[i % len(color_cycle)] for i, exp in enumerate(experiments)}

    line_styles = ["-", "--", ":", "-."]
    schedule_styles = {sched: line_styles[i % len(line_styles)] for i, sched in enumerate(schedules)}

    plt.figure(figsize=(20, 14))
    plt.style.use("seaborn-v0_8-whitegrid")

    plotted_any = False
    for exp in experiments:
        for sched in schedules:
            epoch_to_stats = aggregated[exp].get(sched)
            if not epoch_to_stats:
                continue
            epochs, means, _ = _epochs_and_stats(epoch_to_stats)
            if not epochs:
                continue
            plt.plot(
                epochs,
                means,
                color=exp_colors[exp],
                linestyle=schedule_styles[sched],
                marker="o",
                markersize=4,
                linewidth=1.75,
            )
            plotted_any = True

    if not plotted_any:
        print("Error: No numeric epoch milestones found across any experiment.")
        plt.close()
        return

    plt.title(
        "FID vs. Training Epoch, All Training Distributions x Sampling Schedules "
        "(mean across seeds)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Training Epoch", fontsize=13)
    plt.ylabel("FID Score (Lower is Better)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.7)

    color_handles = [
        Line2D([0], [0], color=exp_colors[exp], linewidth=2, label=exp) for exp in experiments
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle=schedule_styles[sched], linewidth=2, label=sched)
        for sched in schedules
    ]
    color_legend = plt.legend(
        handles=color_handles,
        title="Training Distribution",
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    plt.gca().add_artist(color_legend)
    plt.legend(
        handles=style_handles,
        title="Sampling Schedule",
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
    )

    plt.subplots_adjust(left=0.06, right=0.68, top=0.95, bottom=0.06)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Combined plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed FID results (master_fid_results.json) into "
        "mean/std per (train_dist, schedule, epoch), and optionally plot them."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="results/master_fid_results.json",
        help="Path to the master per-seed FID results file from evaluate_fid.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/aggregated_fid_results.json",
        help="Where to write the seed-aggregated (mean/std) results",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="If set, plot one curve per sampling schedule for this base training "
        "distribution (e.g. 'uniform', 'logit_normal_mu_0.0_sigma_1.0').",
    )
    parser.add_argument(
        "--compare_schedule",
        type=str,
        default=None,
        help="If set, plot one curve per base training distribution for this single "
        "fixed sampling schedule (e.g. 'uniform', 'shifted', 'logit_normal').",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="If set, plot every (training distribution, sampling schedule) curve "
        "together on one figure (means only, no error bands).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="fid_curves_aggregated.png",
        help="Where to save the generated plot (only used with --experiment, "
        "--compare_schedule, or --combined)",
    )
    args = parser.parse_args()

    try:
        with open(args.json_file, "r") as f:
            master_data = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: '{args.json_file}' not found. Make sure to run evaluate_fid.py first!"
        )
        return

    aggregated = aggregate(master_data)

    with open(args.output, "w") as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated results written to {args.output}")
    print(f"Base experiments found: {list(aggregated.keys())}")

    if args.combined:
        plot_all_combined(aggregated, args.save_path)
    elif args.compare_schedule is not None:
        plot_schedule_comparison(aggregated, args.compare_schedule, args.save_path)
    elif args.experiment is not None:
        plot_experiment(aggregated, args.experiment, args.save_path)


if __name__ == "__main__":
    main()
