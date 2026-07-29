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
SCHEDULES = ["uniform", "logit_normal", "shifted_fine", "shifted_coarse"]


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
    args = parser.parse_args()

    span, step = args.epochs.split(":")
    start, end = span.split("-")
    epochs = list(range(int(start), int(end) + 1, int(step)))

    with open(args.results) as f:
        results = json.load(f)

    unscored, no_samples, no_run, scored = [], [], [], 0
    retrain_idx, reeval_idx = set(), set()

    for dist_idx, (train_dist, dist_params) in enumerate(DIST_SPECS):
        for seed in range(args.seeds):
            array_idx = dist_idx * args.seeds + seed
            exp_name = make_exp_name(
                args.dataset, args.conditioning, train_dist, dist_params, seed
            )
            exp_dir = os.path.join(args.eval_runs_dir, exp_name)
            if not os.path.isdir(exp_dir):
                no_run.append((array_idx, dist_idx, seed, exp_name))
                retrain_idx.add(array_idx)
                continue
            for epoch in epochs:
                for schedule in SCHEDULES:
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

    total = len(DIST_SPECS) * args.seeds * len(epochs) * len(SCHEDULES)
    print(f"=== {args.dataset} / cond={args.conditioning} ({total} cells expected) ===")
    print(f"  scored     {scored:>6}")
    print(f"  UNSCORED   {len(unscored):>6}  <- re-run the eval array only")
    print(f"  NO SAMPLES {len(no_samples):>6}  <- epoch never exported; retrain")
    print(f"  NO RUN     {len(no_run) * len(epochs) * len(SCHEDULES):>6}"
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


if __name__ == "__main__":
    main()
