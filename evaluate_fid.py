# written by CLAUDE

import glob
import json
import os
from pathlib import Path

import torch
from cleanfid import fid

# FID here is a one-time offline pass over ~1000 small images, not a training
# bottleneck -- run it on CPU so it doesn't depend on matching cleanfid's
# torch build to whichever GPU node/driver version this happens to run on
# (the cluster's partitions span very different driver versions).
DEVICE = torch.device("cpu")


def run_evaluation():
    REAL_DIR = "data/real"  # TODO update this before running
    EVAL_RUNS_DIR = "eval_runs"
    MASTER_METRICS_FILE = "master_fid_results.json"

    if not os.path.exists(REAL_DIR):
        raise FileNotFoundError(
            f"Could not find your real images directory at: {REAL_DIR}"
        )

    print("==================================================")
    # Find all leaf dirs that look like: eval_runs/experiment_name/epoch_X/schedule_name
    # (NOT epoch_X itself -- clean-fid globs images recursively, so scoring at the
    # epoch level would silently merge all sampling schedules into one FID number.)
    schedule_dirs = glob.glob(os.path.join(EVAL_RUNS_DIR, "*", "epoch_*", "*"))
    schedule_dirs = [d for d in schedule_dirs if os.path.isdir(d)]

    if not schedule_dirs:
        print("No evaluation directories found in 'eval_runs/'.")
        return

    # results[experiment_name][schedule_name][epoch_str] = fid_score
    results = {}

    for schedule_dir in sorted(schedule_dirs):
        path_parts = Path(schedule_dir).parts
        experiment_name = path_parts[1]  # e.g., 'uniform'
        epoch_str = path_parts[2]  # e.g., 'epoch_10'
        schedule_name = path_parts[3]  # e.g., 'uniform', 'shifted', 'logit_normal'

        if not os.listdir(schedule_dir):
            print(f"  -> [SKIP] {schedule_dir} is empty")
            continue

        print(
            f"Calculating FID for {experiment_name} / {schedule_name} ({epoch_str})..."
        )

        try:
            # clean-fid automatically handles image processing and computes the score
            score = fid.compute_fid(REAL_DIR, schedule_dir, device=DEVICE)

            results.setdefault(experiment_name, {}).setdefault(schedule_name, {})
            results[experiment_name][schedule_name][epoch_str] = round(float(score), 4)
            print(f"  -> FID: {score:.4f}")

        except Exception as e:
            print(f"  -> [ERROR] Failed evaluating {schedule_dir}: {e}")

    # Write all metrics out to a single master file for easy plotting later
    with open(MASTER_METRICS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("==================================================")
    print(f"Evaluation complete! Master results saved to {MASTER_METRICS_FILE}")


if __name__ == "__main__":
    run_evaluation()
