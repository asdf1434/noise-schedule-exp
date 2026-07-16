# WRITTEN BY CLAUDE

import glob
import json


def deep_merge(a: dict, b: dict) -> dict:
    for key, value in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(value, dict):
            deep_merge(a[key], value)
        else:
            a[key] = value
    return a


SHARD_DIR = "results/fid_shards"


def main():
    shard_files = sorted(glob.glob(f"{SHARD_DIR}/master_fid_results_shard*.json"))
    if not shard_files:
        print(f"No shard files found ({SHARD_DIR}/master_fid_results_shard*.json).")
        return

    merged = {}
    for path in shard_files:
        with open(path) as f:
            deep_merge(merged, json.load(f))

    with open("results/master_fid_results.json", "w") as f:
        json.dump(merged, f, indent=4)

    n_entries = sum(len(sched) for exp in merged.values() for sched in exp.values())
    print(
        f"Merged {len(shard_files)} shard files ({n_entries} total entries) "
        "-> results/master_fid_results.json"
    )


if __name__ == "__main__":
    main()
