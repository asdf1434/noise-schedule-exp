"""List failed Slurm array task IDs from logs/slurm/*.out, for requeue_failed.sh.

Shares its pass/fail rule with check_slurm_logs.py via log_status.py: a task is
OK only if it printed its script's completion marker. Tasks still running (log
touched recently) are never listed, so a requeue can't duplicate live work.

Pass --expect N to also list array indices that produced no log at all -- tasks
that never started are otherwise invisible and get silently dropped from a sweep.

Prints a Slurm --array-compatible comma-separated list of task IDs (e.g.
"3,7,12") to stdout and nothing else, so it can be captured directly by a shell
script. Prints an empty line if no failures are found.

Usage: python scripts/monitor/list_failed_tasks.py "<pattern>" [--expect N]
Example: python scripts/monitor/list_failed_tasks.py "slurm_exp1_pilot_1097429_*.out"
"""

import argparse

from log_status import FAILED, scan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pattern",
        help="glob pattern relative to logs/slurm/, e.g. 'slurm_exp1_pilot_1097429_*.out'",
    )
    parser.add_argument(
        "--expect",
        type=int,
        help="array size; also lists indices in 0..N-1 that produced no log",
    )
    parser.add_argument("--running_window", type=float, default=30.0)
    args = parser.parse_args()

    results, missing = scan(args.pattern, args.expect, args.running_window)

    ids = [k for k, (s, _, _) in results.items() if s == FAILED and isinstance(k, int)]
    print(",".join(str(i) for i in sorted(set(ids) | set(missing))))


if __name__ == "__main__":
    main()
