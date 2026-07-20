# written by CLAUDE

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid

from src.naming import parse_exp_name

# FID eval takes forever on cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 16
BATCH_SIZE = 128

REAL_STATS_NAME = {
    "mnist": "mnist_real",
    "eurosat": "eurosat_real",
}
REAL_DIR = {
    "mnist": "data/real",
    "eurosat": "data/real_eurosat",
}
EVAL_RUNS_DIR = "eval_runs"
SHARD_DIR = "results/fid_shards"


def _epoch_sort_key(schedule_dir: str) -> int:
    match = re.search(r"epoch_(\d+)", schedule_dir)
    return int(match.group(1)) if match else -1


def _belongs_to_dataset(experiment_name: str, dataset: str) -> bool:
    try:
        return parse_exp_name(experiment_name)["dataset"] == dataset
    except ValueError:
        # Pre-rename experiments (exp1/2/3) predate the ds-/cond-/dist-/seed-
        # naming scheme and were never migrated; fall back to the old
        # convention where non-mnist datasets prefixed the bare name, e.g.
        # eurosat_uniform_seed42, and mnist experiments had no prefix.
        if dataset == "mnist":
            return not experiment_name.startswith("eurosat_")
        return experiment_name.startswith(f"{dataset}_")


def run_evaluation(
    shard: int, num_shards: int, dataset: str = "mnist", allow_cpu: bool = False
):
    if DEVICE.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "No GPU visible (torch.cuda.is_available() is False) and --allow_cpu wasn't passed. Refusing to silently fall back onto a CPU run."
        )

    # each shard writes to its own file so tasks never try to write to the master file at the same time
    # merge_fid_shards.py combines these files afterward.

    if num_shards == 1:
        metrics_file = "results/master_fid_results.json"
    else:
        os.makedirs(SHARD_DIR, exist_ok=True)
        metrics_file = os.path.join(
            SHARD_DIR, f"master_fid_results_shard{shard}_{dataset}.json"
        )

    real_dir = REAL_DIR[dataset]
    real_stats_name = REAL_STATS_NAME[dataset]

    if not os.path.exists(real_dir):
        raise FileNotFoundError(
            f"Could not find your real images directory at: {real_dir}"
        )

    if not fid.test_stats_exists(real_stats_name, mode="clean"):
        raise RuntimeError(
            f"Real-image FID stats '{real_stats_name}' aren't cached yet. Run "
            f"cache_real_stats.py --dataset {dataset} once first (see run_exp1_eval_stats.sh)."
        )
    # this gets built once per process on the real data
    feat_model = fid.build_feature_extractor("clean", DEVICE)
    ref_mu, ref_sigma = fid.get_reference_statistics(
        real_stats_name, res="na", mode="clean", split="custom"
    )

    print("==================================================")
    # find all leaf directories that look like eval_runs/experiment_name/epoch_X/schedule_name
    schedule_dirs = glob.glob(os.path.join(EVAL_RUNS_DIR, "*", "epoch_*", "*"))
    schedule_dirs = [d for d in schedule_dirs if os.path.isdir(d)]
    schedule_dirs = [
        d for d in schedule_dirs if _belongs_to_dataset(Path(d).parts[1], dataset)
    ]
    schedule_dirs = sorted(schedule_dirs, key=_epoch_sort_key)

    if not schedule_dirs:
        print(f"No evaluation directories found in 'eval_runs/' for dataset={dataset}.")
        return

    # results[experiment_name][schedule_name][epoch_str] = fid_score
    # Resume from a prior (possibly time-limit-killed) run instead of
    # recomputing folders that were already scored.
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            results = json.load(f)
        print(f"Resuming from existing {metrics_file}")
    else:
        results = {}

    pending_dirs = []
    for schedule_dir in schedule_dirs:
        path_parts = Path(schedule_dir).parts
        experiment_name = path_parts[1]
        epoch_str = path_parts[2]
        schedule_name = path_parts[3]
        if epoch_str in results.get(experiment_name, {}).get(schedule_name, {}):
            continue
        pending_dirs.append(schedule_dir)

    # Give this shard every num_shards-th pending folder, so the whole sweep
    # (across all array tasks) covers every folder exactly once.
    my_dirs = pending_dirs[shard::num_shards]
    print(
        f"Shard {shard}/{num_shards}: scoring {len(my_dirs)} of "
        f"{len(pending_dirs)} pending folders"
    )

    for schedule_dir in my_dirs:
        path_parts = Path(schedule_dir).parts
        experiment_name = path_parts[1]
        epoch_str = path_parts[2]
        schedule_name = path_parts[3]

        if not os.listdir(schedule_dir):
            print(f"  -> [SKIP] {schedule_dir} is empty")
            continue

        print(
            f"Calculating FID for {experiment_name} / {schedule_name} ({epoch_str})..."
        )

        try:
            # scored against the cached real-image stats and the resident
            # feature model built above, not rebuilt/reloaded per folder
            feats = fid.get_folder_features(
                schedule_dir,
                model=feat_model,
                num_workers=NUM_WORKERS,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                mode="clean",
                verbose=True,
            )
            mu = np.mean(feats, axis=0)
            sigma = np.cov(feats, rowvar=False)
            score = fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)

            results.setdefault(experiment_name, {}).setdefault(schedule_name, {})
            results[experiment_name][schedule_name][epoch_str] = round(float(score), 4)
            print(f"  -> FID: {score:.4f}")

            # Write after every folder (not just at the end) so a time-limit
            # kill or crash mid-sweep doesn't lose already-computed scores.
            with open(metrics_file, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"  -> [ERROR] Failed evaluating {schedule_dir}: {e}")

    print("==================================================")
    print(f"Shard {shard}/{num_shards} complete! Results saved to {metrics_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This task's index (0-based) among num_shards",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of parallel shards/tasks",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="allow running on cpu even if no gpu?",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "eurosat"],
        help="which dataset's eval_runs/ experiments to score (run once per dataset)",
    )
    args = parser.parse_args()
    run_evaluation(
        args.shard, args.num_shards, dataset=args.dataset, allow_cpu=args.allow_cpu
    )


if __name__ == "__main__":
    main()
