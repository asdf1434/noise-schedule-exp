"""Generic error/progress scan over logs/slurm/*.out, for any job (training
array, eval array, merge, etc) -- not tied to one specific experiment.

A task is OK only if it printed its script's completion marker; everything else
is a failure (see log_status.py for why it's an allowlist of success rather
than of errors). Tasks that never started write no log at all, so pass
--expect N to have their array indices reported as MISSING.

For each matched log file, reports status, the node it ran on (so repeat
failures on one node are easy to spot), and the line explaining the failure.

Exits 1 if anything failed or is missing, so it can gate a pipeline stage.

Usage:
  python check_slurm_logs.py                            # scan everything
  python check_slurm_logs.py "*exp2_train_dists*"        # just exp2 training
  python check_slurm_logs.py "*1216394*" --expect 120    # flag tasks with no log
"""

import argparse
import sys
from collections import Counter

from log_status import FAILED, OK, RUNNING, scan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*.out",
        help="glob pattern (relative to logs/slurm/) to scan, e.g. '*eval_array*'",
    )
    parser.add_argument(
        "--expect",
        type=int,
        help="array size (e.g. 120); reports indices in 0..N-1 that produced no log",
    )
    parser.add_argument(
        "--running_window",
        type=float,
        default=30.0,
        help="a marker-less log touched within this many minutes counts as running",
    )
    args = parser.parse_args()

    results, missing = scan(args.pattern, args.expect, args.running_window)

    if not results and not missing:
        print(f"No logs found matching logs/slurm/{args.pattern}")
        return 0

    buckets = Counter(status for status, _, _ in results.values())
    print(f"Scanned {len(results)} log(s) matching '{args.pattern}'\n")
    print(f"{buckets[OK]} completed cleanly")
    print(f"{buckets[RUNNING]} still running")
    print(f"{buckets[FAILED]} FAILED")
    if args.expect is not None:
        print(f"{len(missing)} MISSING (never wrote a log)")
    print()

    failed = sorted(
        (k for k, (s, _, _) in results.items() if s == FAILED),
        key=lambda k: (isinstance(k, str), k),
    )
    if failed:
        print("Failing tasks (task -> node -> why):")
        for key in failed:
            _, node, reason = results[key]
            print(f"  task {str(key):>4}  {node:<20}  {reason}")

        node_counts = Counter(results[k][1] for k in failed)
        print("\nFailures by node (repeat offenders worth excluding via --exclude):")
        for node, count in node_counts.most_common():
            print(f"  {node:<20} {count} failure(s)")

    if missing:
        print(f"\nMISSING -- no log at all, so these never started ({len(missing)}):")
        print(f"  --array={','.join(str(i) for i in missing)}")

    if failed or missing:
        print("\nRequeue everything above with:")
        print(f'  scripts/monitor/requeue_failed.sh "{args.pattern}" <script.sh>'
              + (f" {args.expect}" if args.expect is not None else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
