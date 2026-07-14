"""Generic error/progress scan over logs/slurm/*.out, for any job (training
array, eval array, merge, etc) -- not tied to one specific experiment.

For each matched log file, reports:
  - whether it contains an error (Error/Traceback/CANCELLED/error:)
  - whether it reached a "complete" marker printed by the corresponding
    .sh script (e.g. "Array task N ... complete", "Shard N/M complete")
  - which node it ran on (from the cpu-bind line), so repeat failures on
    the same node are easy to spot

Usage:
  python check_slurm_logs.py                          # scan everything
  python check_slurm_logs.py "*exp2_train_dists*"      # just exp2 training
  python check_slurm_logs.py "*eval_array*"            # just the eval array
"""

import argparse
import glob
import os
import re
from collections import Counter, defaultdict

LOG_DIR = "logs/slurm"

ERROR_MARKERS = ("Error", "Traceback", "CANCELLED", "error:")
COMPLETE_MARKERS = ("complete.", "complete!")

NODE_RE = re.compile(r"cpu-bind=MASK - ([\w.\-]+),")
TASK_ID_RE = re.compile(r"_(\d+)\.out$")


def scan(pattern: str):
    paths = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    if not paths:
        print(f"No logs found matching {os.path.join(LOG_DIR, pattern)}")
        return

    errored = []
    completed = []
    incomplete = []
    node_of = {}
    error_node_counts = Counter()

    for path in paths:
        try:
            with open(path, errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        node_match = NODE_RE.search(text)
        node = node_match.group(1) if node_match else "?"
        node_of[path] = node

        has_error = any(marker in text for marker in ERROR_MARKERS)
        has_complete = any(marker in text for marker in COMPLETE_MARKERS)

        if has_error:
            errored.append(path)
            error_node_counts[node] += 1
        elif has_complete:
            completed.append(path)
        else:
            incomplete.append(path)  # still running, or died with no marker at all

    print(f"Scanned {len(paths)} log(s) matching '{pattern}'\n")

    print(f"{len(completed)} completed cleanly")
    print(f"{len(incomplete)} still running / no completion marker yet")
    print(f"{len(errored)} contain an error\n")

    if errored:
        print("Failing tasks (path -> node):")
        for path in errored:
            task_id = TASK_ID_RE.search(path)
            task_id = task_id.group(1) if task_id else "?"
            print(f"  task {task_id:>4}  {node_of[path]:<20}  {path}")

        print("\nFailures by node (repeat offenders worth excluding via --exclude):")
        for node, count in error_node_counts.most_common():
            print(f"  {node:<20} {count} failure(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*.out",
        help="glob pattern (relative to logs/slurm/) to scan, e.g. '*eval_array*'",
    )
    args = parser.parse_args()
    scan(args.pattern)


if __name__ == "__main__":
    main()
