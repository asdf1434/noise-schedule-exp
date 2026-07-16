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

# FID evaluation is the pipeline bottleneck, so prefer GPU when the Slurm
# allocation provides one (see scripts/slurm/run_exp1_eval_array.sh's
# --gres=gpu:1). Falls back to CPU so a plain `python evaluate_fid.py` still
# works on a CPU-only box.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
BATCH_SIZE = 128

REAL_STATS_NAME = "mnist_real"
REAL_DIR = "data/real"
EVAL_RUNS_DIR = "eval_runs"
SHARD_DIR = "results/fid_shards"


def _epoch_sort_key(schedule_dir: str) -> int:
    match = re.search(r"epoch_(\d+)", schedule_dir)
    return int(match.group(1)) if match else -1


def run_evaluation(shard: int, num_shards: int):
    # When run under the Slurm array (run_exp1_eval_array.sh), each shard
    # writes its own file so N concurrent array tasks never write to the
    # same master file at once. merge_fid_shards.py combines them afterward.
    # With num_shards=1 (plain `python evaluate_fid.py`), this is just
    # master_fid_results.json directly.
    if num_shards == 1:
        metrics_file = "results/master_fid_results.json"
    else:
        os.makedirs(SHARD_DIR, exist_ok=True)
        metrics_file = os.path.join(SHARD_DIR, f"master_fid_results_shard{shard}.json")

    if not os.path.exists(REAL_DIR):
        raise FileNotFoundError(f"Could not find your real images directory at: {REAL_DIR}")

    if not fid.test_stats_exists(REAL_STATS_NAME, mode="clean"):
        raise RuntimeError(
            f"Real-image FID stats '{REAL_STATS_NAME}' aren't cached yet. Run "
            "cache_real_stats.py once first (see run_exp1_eval_stats.sh)."
        )

    # Built once per process (not per folder) -- rebuilding the InceptionV3
    # feature extractor and re-downloading/re-parsing reference stats per
    # folder would dominate GPU wall-clock for these small (~1000-image) sets.
    feat_model = fid.build_feature_extractor("clean", DEVICE)
    ref_mu, ref_sigma = fid.get_reference_statistics(
        REAL_STATS_NAME, res="na", mode="clean", split="custom"
    )

    print("==================================================")
    # Find all leaf dirs that look like: eval_runs/experiment_name/epoch_X/schedule_name
    # (NOT epoch_X itself -- clean-fid globs images recursively, so scoring at the
    # epoch level would silently merge all sampling schedules into one FID number.)
    schedule_dirs = glob.glob(os.path.join(EVAL_RUNS_DIR, "*", "epoch_*", "*"))
    schedule_dirs = [d for d in schedule_dirs if os.path.isdir(d)]
    schedule_dirs = sorted(schedule_dirs, key=_epoch_sort_key)

    if not schedule_dirs:
        print("No evaluation directories found in 'eval_runs/'.")
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

        print(f"Calculating FID for {experiment_name} / {schedule_name} ({epoch_str})...")

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
        "--shard", type=int, default=0, help="This task's index (0-based) among num_shards"
    )
    parser.add_argument(
        "--num_shards", type=int, default=1, help="Total number of parallel shards/tasks"
    )
    args = parser.parse_args()
    run_evaluation(args.shard, args.num_shards)


if __name__ == "__main__":
    main()
