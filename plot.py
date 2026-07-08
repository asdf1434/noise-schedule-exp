# WRITTEN BY CLAUDE

import argparse
import json

import matplotlib.pyplot as plt


def _epochs_and_scores(epoch_to_score: dict) -> tuple[list, list]:
    """Parses {'epoch_10': 23.4, ...} into sorted (epochs, scores) lists."""
    epochs, scores = [], []
    for epoch_key, score in epoch_to_score.items():
        numeric_epoch = "".join(c for c in epoch_key if c.isdigit())
        if numeric_epoch:
            epochs.append(int(numeric_epoch))
            scores.append(score)
    sorted_pairs = sorted(zip(epochs, scores))
    if not sorted_pairs:
        return [], []
    sorted_epochs, sorted_scores = zip(*sorted_pairs)
    return list(sorted_epochs), list(sorted_scores)


def plot_experiment(master_data: dict, experiment: str, save_path: str):
    """One figure, one curve per sampling schedule, for a single training-dist experiment.

    This is the core Q1 deliverable: for a fixed training noise-level distribution,
    compare FID across the different inference-time sampling sequences that were
    evaluated at each checkpoint.
    """
    if experiment not in master_data:
        print(f"Error: Experiment '{experiment}' not found in results.")
        print(f"Available tracking runs: {list(master_data.keys())}")
        return

    schedule_data = master_data[experiment]

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plotted_any = False
    for schedule_name in sorted(schedule_data.keys()):
        epochs, scores = _epochs_and_scores(schedule_data[schedule_name])
        if not epochs:
            continue
        plt.plot(epochs, scores, marker="o", linewidth=2, label=schedule_name)
        plotted_any = True

    if not plotted_any:
        print(f"Error: No numeric epoch milestones found for '{experiment}'.")
        plt.close()
        return

    plt.title(
        f"FID vs. Training Epoch by Sampling Schedule (Train Dist: {experiment})",
        fontsize=15,
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


def plot_schedule_comparison(master_data: dict, schedule: str, save_path: str):
    """One figure, one curve per training-dist experiment, for a single fixed
    sampling schedule -- lets you compare training distributions head-to-head.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plotted_any = False
    for experiment in sorted(master_data.keys()):
        epoch_to_score = master_data[experiment].get(schedule)
        if not epoch_to_score:
            continue
        epochs, scores = _epochs_and_scores(epoch_to_score)
        if not epochs:
            continue
        plt.plot(epochs, scores, marker="o", linewidth=2, label=experiment)
        plotted_any = True

    if not plotted_any:
        print(f"Error: No experiments have data for sampling schedule '{schedule}'.")
        plt.close()
        return

    plt.title(
        f"FID vs. Training Epoch by Training Distribution (Sampling: {schedule})",
        fontsize=15,
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot FID curves from master offline JSON metrics."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="master_fid_results.json",
        help="Path to the master consolidated FID results file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="uniform",
        help="Name of the training experiment run to isolate and plot (e.g., 'uniform', 'logit_normal'). "
        "Plots one curve per sampling schedule for this training distribution.",
    )
    parser.add_argument(
        "--compare_schedule",
        type=str,
        default=None,
        help="If set, ignores --experiment and instead plots one curve per training "
        "distribution for this single fixed sampling schedule (e.g. 'uniform', 'shifted', 'logit_normal').",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="fid_curves.png",
        help="Where to save the generated plot",
    )
    args = parser.parse_args()

    # Load the consolidated evaluation data
    try:
        with open(args.json_file, "r") as f:
            master_data = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: Master file '{args.json_file}' not found. Make sure to run evaluate_fid.py first!"
        )
        return

    if args.compare_schedule is not None:
        plot_schedule_comparison(master_data, args.compare_schedule, args.save_path)
    else:
        plot_experiment(master_data, args.experiment, args.save_path)


if __name__ == "__main__":
    main()
