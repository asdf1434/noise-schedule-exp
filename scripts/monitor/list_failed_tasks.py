"""List failed Slurm array task IDs from logs/slurm/*.out, for requeue_failed.sh.

Reuses the same error detection as check_slurm_logs.py (Error/Traceback/
CANCELLED/error: anywhere in the log). Prints a Slurm --array-compatible
comma-separated list of task IDs (e.g. "3,7,12") to stdout and nothing else,
so it can be captured directly by a shell script. Prints an empty line if
no failures are found.

Usage: python scripts/list_failed_tasks.py "<pattern>"
Example: python scripts/list_failed_tasks.py "slurm_exp1_pilot_1097429_*.out"
"""

import argparse
import glob
import os
import re

LOG_DIR = "logs/slurm"
ERROR_MARKERS = ("Error", "Traceback", "CANCELLED", "error:")
TASK_ID_RE = re.compile(r"_(\d+)\.out$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="glob pattern relative to logs/slurm/, e.g. 'slurm_exp1_pilot_1097429_*.out'")
    args = parser.parse_args()

    failed_ids = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, args.pattern))):
        try:
            with open(path, errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        if any(marker in text for marker in ERROR_MARKERS):
            match = TASK_ID_RE.search(path)
            if match:
                failed_ids.append(match.group(1))

    print(",".join(failed_ids))


if __name__ == "__main__":
    main()
