# Prime every FID shard file with what's already in results/master_fid_results.json.
#
# evaluate_fid.py resumes from its OWN shard file only, so naively re-submitting an
# eval array after some shards died makes each surviving shard treat every folder it
# didn't personally score as pending -- lots of duplicated work. Seeding all shard
# files with the merged master first means pending_dirs is exactly the set of folders
# nobody has scored yet, and the stride slice partitions that set cleanly.
#
# Usage (on the cluster, from repo root, BEFORE re-submitting run_*_eval_array.sh):
#   python scripts/monitor/seed_fid_shards.py --dataset mnist --num_shards 64

import argparse
import json
import os

SHARD_DIR = "results/fid_shards"
MASTER = "results/master_fid_results.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_shards", type=int, default=64)
    args = parser.parse_args()

    with open(MASTER) as f:
        master = json.load(f)

    os.makedirs(SHARD_DIR, exist_ok=True)
    for shard in range(args.num_shards):
        path = os.path.join(
            SHARD_DIR, f"master_fid_results_shard{shard}_{args.dataset}.json"
        )
        with open(path, "w") as f:
            json.dump(master, f, indent=4)

    n = sum(len(sched) for exp in master.values() for sched in exp.values())
    print(f"Seeded {args.num_shards} shard files for {args.dataset} with {n} entries.")


if __name__ == "__main__":
    main()
