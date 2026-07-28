# WRITTEN BY CLAUDE

"""Splits results/master_fid_results.json into one file per training run
(one file per experiment_name key -- dataset x conditioning x train_dist x
seed for canonical names, or whatever the legacy pre-rename name was),
written to results/fid_by_run/<ds-...__cond-...>/<dist-...__seed-...>.json.

Canonical names (see src/naming.py) are split into a subfolder per
dataset+conditioning ("ds-{dataset}__cond-{conditioning}") with one file per
dist+seed inside it. Legacy pre-rename names (no ds-/cond-/dist-/seed-
fields) don't have a dataset/conditioning to group by, so they're written
flat into a "legacy" subfolder instead.

Existing tools (aggregate_fid.py, plot.py, evaluate_fid.py's own writes) keep
reading/writing the single master_fid_results.json -- this is a read-only
export for browsing/diffing individual runs without grepping through one
~700-entry file.
"""

import argparse
import json
import os

from src.naming import parse_exp_name

MASTER_PATH = "results/master_fid_results.json"
OUTPUT_DIR = "results/fid_by_run"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json_file", type=str, default=MASTER_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    with open(args.json_file) as f:
        master_data = json.load(f)

    for experiment_name, schedule_data in master_data.items():
        try:
            parsed = parse_exp_name(experiment_name)
            run_dir = f"ds-{parsed['dataset']}__cond-{parsed['conditioning']}"
            file_name = f"dist-{parsed['train_dist_full']}__seed-{parsed['seed']}.json"
        except ValueError:
            run_dir = "legacy"
            file_name = f"{experiment_name}.json"

        out_dir = os.path.join(args.output_dir, run_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, file_name), "w") as f:
            json.dump(schedule_data, f, indent=4)

    print(f"Wrote {len(master_data)} per-run files under {args.output_dir}/")


if __name__ == "__main__":
    main()
