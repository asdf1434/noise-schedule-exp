"""Diff the sweep grid against what's actually on disk, so you can tell whether a
gap needs a *retrain* or only a *re-eval*.

Run this on the cluster (it needs eval_runs/ next to results/master_fid_results.json):

    python scripts/monitor/check_missing.py --dataset mnist --conditioning inpaint

For every (train_dist, seed, epoch, sampling_schedule) cell in the sweep it reports one of:

  scored      FID is already in results/master_fid_results.json -- nothing to do
  UNSCORED    eval_runs/ folder has its images but no FID -- re-run the eval array only
  NO SAMPLES  folder missing/short -- that epoch never got exported, so retrain the run
  NO RUN      no eval_runs/<exp_name>/ at all -- retrain the run

and prints ready-to-paste `sbatch --array=` indices for whatever needs retraining.
`--list_unscored FILE` dumps the UNSCORED folder paths one per line.

Defaults describe the pre-2026-08 sweeps (6 training distributions, 4 sampling
schedules). For the newer sweeps pass `--dists neutral3 --schedules all --seeds 10`,
plus `--array_offset` when several datasets share one job array -- otherwise the
counts and the printed indices will both be wrong.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from src.naming import make_exp_name  # noqa: E402

# Order matters: array index == dist_idx * 20 + seed, matching scripts/slurm/run_*.sh
DIST_SPECS = [
    ("uniform", {}),
    ("logit_normal", {"mu": 0.0, "sigma": 1.0}),
    ("logit_normal", {"mu": 0.0, "sigma": 0.3}),
    ("logit_normal", {"mu": 1.5, "sigma": 1.0}),
    ("logit_normal", {"mu": -1.5, "sigma": 1.0}),
    ("plateau_logit_normal", {"mu": 0.0, "sigma": 1.0, "uniform_prob": 0.3}),
]
DIST_LABELS = [
    "uniform",
    "logit_normal",
    "logit_normal_peaked(sigma=0.3)",
    "logit_normal_skew_hi(mu=+1.5)",
    "logit_normal_skew_lo(mu=-1.5)",
    "plateau",
]
# The four original spacings, present in every sweep ever run.
SCHEDULES_LEGACY = ["uniform", "logit_normal", "shifted_fine", "shifted_coarse"]
# Runs from the 2026-08 sweeps onward also export these five extra shift values
# (see schedules_to_test in train.py). Older runs never had them, so checking the
# full list against a pre-August sweep would report thousands of phantom gaps --
# hence --schedules, defaulting to the legacy set.
SCHEDULES_EXTRA = [
    "shifted_s0.15",
    "shifted_s0.5",
    "shifted_s0.7",
    "shifted_s1.5",
    "shifted_s5.0",
]
# The 2026-08 sweeps drop the three distributions that earlier results settled
# (both skews, plus one of the interchangeable neutral ones) and keep only these,
# in this order -- see scripts/slurm/run_fashion_sizes.sh and friends.
NEUTRAL3_IDX = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--conditioning", default="none")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--epochs", default="10-100:10", help="start-end:step")
    parser.add_argument("--eval_runs_dir", default="eval_runs")
    parser.add_argument("--results", default="results/master_fid_results.json")
    parser.add_argument(
        "--min_images",
        type=int,
        default=1000,
        help="a schedule folder with fewer files than this counts as not exported",
    )
    parser.add_argument("--list_unscored", help="write UNSCORED folder paths here")
    parser.add_argument(
        "--schedules",
        choices=["legacy", "all"],
        default="legacy",
        help="'legacy' = the original 4 spacings (use for any pre-2026-08 sweep); "
        "'all' = those plus the 5 extra shift values that newer runs export",
    )
    parser.add_argument(
        "--dists",
        choices=["all", "neutral3"],
        default="all",
        help="'all' = the original 6 training distributions; 'neutral3' = the 3 kept by the "
        "2026-08 sweeps (uniform, logit_normal, logit_normal_peaked)",
    )
    parser.add_argument(
        "--array_offset",
        type=int,
        default=0,
        help="added to the printed sbatch indices, for arrays that pack several "
        "datasets/conditionings into one job (index = block*seeds*dists + dist*seeds + seed)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any cell is unscored or missing, so a pipeline stage can gate on it",
    )
    args = parser.parse_args()

    schedules = SCHEDULES_LEGACY + (SCHEDULES_EXTRA if args.schedules == "all" else [])
    dist_idxs = NEUTRAL3_IDX if args.dists == "neutral3" else list(range(len(DIST_SPECS)))

    span, step = args.epochs.split(":")
    start, end = span.split("-")
    epochs = list(range(int(start), int(end) + 1, int(step)))

    with open(args.results) as f:
        results = json.load(f)

    unscored, no_samples, no_run, scored = [], [], [], 0
    retrain_idx, reeval_idx = set(), set()

    for slot, dist_idx in enumerate(dist_idxs):
        train_dist, dist_params = DIST_SPECS[dist_idx]
        for seed in range(args.seeds):
            array_idx = args.array_offset + slot * args.seeds + seed
            exp_name = make_exp_name(
                args.dataset, args.conditioning, train_dist, dist_params, seed
            )
            exp_dir = os.path.join(args.eval_runs_dir, exp_name)
            if not os.path.isdir(exp_dir):
                no_run.append((array_idx, dist_idx, seed, exp_name))
                retrain_idx.add(array_idx)
                continue
            for epoch in epochs:
                for schedule in schedules:
                    if f"epoch_{epoch}" in results.get(exp_name, {}).get(schedule, {}):
                        scored += 1
                        continue
                    folder = os.path.join(exp_dir, f"epoch_{epoch}", schedule)
                    n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
                    if n >= args.min_images:
                        unscored.append(folder)
                        reeval_idx.add(array_idx)
                    else:
                        no_samples.append((folder, n))
                        retrain_idx.add(array_idx)

    total = len(dist_idxs) * args.seeds * len(epochs) * len(schedules)
    print(f"=== {args.dataset} / cond={args.conditioning} ({total} cells expected) ===")
    print(f"  scored     {scored:>6}")
    print(f"  UNSCORED   {len(unscored):>6}  <- re-run the eval array only")
    print(f"  NO SAMPLES {len(no_samples):>6}  <- epoch never exported; retrain")
    print(f"  NO RUN     {len(no_run) * len(epochs) * len(schedules):>6}"
          f"  ({len(no_run)} runs with no eval_runs/ folder; retrain)")

    if no_run:
        print("\nRuns with no eval_runs/ folder:")
        for array_idx, dist_idx, seed, exp_name in sorted(no_run):
            print(f"  [{array_idx:>3}] {DIST_LABELS[dist_idx]:<30} seed {seed}")
    if no_samples:
        print(f"\nEpochs missing samples (showing up to 20 of {len(no_samples)}):")
        for folder, n in sorted(no_samples)[:20]:
            print(f"  {folder}  ({n} images)")

    if retrain_idx:
        print("\nRETRAIN these array indices:")
        print(f"  --array={','.join(str(i) for i in sorted(retrain_idx))}")
    if unscored:
        print(f"\nRE-EVAL only: {len(unscored)} folders across "
              f"{len(reeval_idx - retrain_idx)} otherwise-complete runs.")
        if args.list_unscored:
            with open(args.list_unscored, "w") as f:
                f.write("\n".join(sorted(unscored)) + "\n")
            print(f"  folder list written to {args.list_unscored}")

    incomplete = len(unscored) + len(no_samples) + len(no_run)
    if not incomplete:
        print("\nComplete: every cell in the grid has a FID score.")
    if args.strict and incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
