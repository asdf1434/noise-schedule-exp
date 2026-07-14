# written by CLAUDE

import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from cleanfid import fid

# FID here is a one-time offline pass over ~1000 small images, not a training
# bottleneck -- run it on CPU so it doesn't depend on matching cleanfid's
# torch build to whichever GPU node/driver version this happens to run on
# (the cluster's partitions span very different driver versions).
DEVICE = torch.device("cpu")
# match the job's actual --cpus-per-task. Folders are scored in parallel
# (one process per folder, see run_evaluation), so each individual
# fid.compute_fid call gets num_workers=0 -- giving DataLoader subprocesses
# of its own here would oversubscribe past what's actually allocated.
POOL_WORKERS = 8

REAL_STATS_NAME = "mnist_real"


def _epoch_sort_key(schedule_dir: str) -> int:
    match = re.search(r"epoch_(\d+)", schedule_dir)
    return int(match.group(1)) if match else -1


def _score_folder(schedule_dir: str):
    """Runs in a worker process: scores one (experiment, epoch, schedule)
    folder against the cached real-image stats. Returns
    (experiment_name, schedule_name, epoch_str, score, error)."""
    path_parts = Path(schedule_dir).parts
    experiment_name = path_parts[1]
    epoch_str = path_parts[2]
    schedule_name = path_parts[3]

    if not os.listdir(schedule_dir):
        return experiment_name, schedule_name, epoch_str, None, "empty"

    try:
        score = fid.compute_fid(
            schedule_dir,
            dataset_name=REAL_STATS_NAME,
            dataset_split="custom",
            mode="clean",
            device=DEVICE,
            num_workers=0,
            verbose=False,
        )
        return experiment_name, schedule_name, epoch_str, float(score), None
    except Exception as e:
        return experiment_name, schedule_name, epoch_str, None, str(e)


def run_evaluation():
    REAL_DIR = "data/real"  # TODO update this before running
    EVAL_RUNS_DIR = "eval_runs"
    MASTER_METRICS_FILE = "master_fid_results.json"

    if not os.path.exists(REAL_DIR):
        raise FileNotFoundError(
            f"Could not find your real images directory at: {REAL_DIR}"
        )

    # Precompute Inception stats for the real reference set ONCE and cache
    # them, instead of recomputing over all real images from scratch for
    # every one of the ~400 (experiment, schedule, epoch) folders below --
    # the real-image stats never change, so that was pure repeated work.
    if not fid.test_stats_exists(REAL_STATS_NAME, mode="clean"):
        print(f"Caching real-image FID stats as '{REAL_STATS_NAME}'...")
        # nothing else is running yet, so this single call can use all
        # allocated CPUs as DataLoader workers
        fid.make_custom_stats(
            REAL_STATS_NAME,
            REAL_DIR,
            mode="clean",
            device=DEVICE,
            num_workers=POOL_WORKERS,
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
    # Resume from a prior (possibly time-limit-killed) run instead of
    # recomputing folders that were already scored.
    if os.path.exists(MASTER_METRICS_FILE):
        with open(MASTER_METRICS_FILE, "r") as f:
            results = json.load(f)
        print(f"Resuming from existing {MASTER_METRICS_FILE}")
    else:
        results = {}

    # Filter out folders already scored (resume) up front, so the pool only
    # ever gets handed real work.
    pending_dirs = []
    for schedule_dir in sorted(schedule_dirs, key=_epoch_sort_key):
        path_parts = Path(schedule_dir).parts
        experiment_name = path_parts[1]
        epoch_str = path_parts[2]
        schedule_name = path_parts[3]
        if epoch_str in results.get(experiment_name, {}).get(schedule_name, {}):
            print(
                f"  -> [SKIP] {experiment_name} / {schedule_name} ({epoch_str}) "
                "already scored"
            )
            continue
        pending_dirs.append(schedule_dir)

    print(f"Scoring {len(pending_dirs)} folders across {POOL_WORKERS} processes...")

    # Folders are independent, so score them in parallel across processes
    # instead of one at a time -- the previous sequential version left
    # every core but one idle for the whole ~3.5 hour sweep.
    with ProcessPoolExecutor(max_workers=POOL_WORKERS) as pool:
        futures = [pool.submit(_score_folder, d) for d in pending_dirs]
        for future in as_completed(futures):
            experiment_name, schedule_name, epoch_str, score, error = future.result()

            if error == "empty":
                print(f"  -> [SKIP] {experiment_name}/{epoch_str}/{schedule_name} is empty")
                continue
            if error is not None:
                print(
                    f"  -> [ERROR] {experiment_name}/{epoch_str}/{schedule_name}: {error}"
                )
                continue

            results.setdefault(experiment_name, {}).setdefault(schedule_name, {})
            results[experiment_name][schedule_name][epoch_str] = round(score, 4)
            print(
                f"  -> FID {experiment_name} / {schedule_name} ({epoch_str}): {score:.4f}"
            )

            # Write after every folder (not just at the end) so a time-limit
            # kill or crash mid-sweep doesn't lose already-computed scores.
            with open(MASTER_METRICS_FILE, "w") as f:
                json.dump(results, f, indent=4)

    print("==================================================")
    print(f"Evaluation complete! Master results saved to {MASTER_METRICS_FILE}")


if __name__ == "__main__":
    run_evaluation()
