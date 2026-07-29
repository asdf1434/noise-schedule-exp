"""Shared pass/fail classification for logs/slurm/*.out, used by
check_slurm_logs.py and list_failed_tasks.py so the two can't disagree.

A task counts as OK only if it printed the completion marker its .sh script
echoes at the end. Anything else is a failure. That inversion matters: the
previous approach scanned for known error strings (Error/Traceback/CANCELLED/
error:), which silently passed native crashes that print none of them, e.g.

    free(): invalid next size (normal)
    /var/lib/slurm/slurmd/.../slurm_script: line 65: 1931546 Aborted (core dumped) python -u train.py ...

That task looked "still running" forever and never got requeued. The set of
ways a job can die is open-ended; the set of ways it can succeed is not.

ERROR_MARKERS are now only used to *explain* a failure in the report, never to
decide whether one happened.
"""

import glob
import os
import re
import time

LOG_DIR = "logs/slurm"

# Printed by the trailing `echo` in every scripts/slurm/*.sh
COMPLETE_MARKERS = ("complete.", "complete!")

# Diagnostic only -- for labelling a failure, not detecting it.
ERROR_MARKERS = (
    "Traceback",
    "Error",
    "error:",
    "CANCELLED",
    "DUE TO TIME LIMIT",
    "Aborted",
    "core dumped",
    "Segmentation fault",
    "free():",
    "double free",
    "std::bad_alloc",
    "Killed",
    "oom-kill",
    "Out of memory",
    "CUDA_ERROR",
    "RESOURCES",
    "NODE_FAIL",
)

NODE_RE = re.compile(r"cpu-bind=MASK - ([\w.\-]+),")
TASK_ID_RE = re.compile(r"_(\d+)\.out$")

OK = "ok"
FAILED = "failed"
RUNNING = "running"


def task_id(path: str):
    match = TASK_ID_RE.search(path)
    return int(match.group(1)) if match else None


def classify(path: str, running_window_min: float = 30.0):
    """Returns (status, node, reason).

    A log with no completion marker that was touched within running_window_min
    is assumed to still be running rather than dead -- otherwise every in-flight
    task would be reported as a failure mid-array.
    """
    try:
        with open(path, errors="ignore") as f:
            text = f.read()
    except OSError as exc:
        return FAILED, "?", f"unreadable log ({exc})"

    node_match = NODE_RE.search(text)
    node = node_match.group(1) if node_match else "?"

    if any(marker in text for marker in COMPLETE_MARKERS):
        return OK, node, ""

    age_min = (time.time() - os.path.getmtime(path)) / 60.0
    if age_min < running_window_min:
        return RUNNING, node, f"no completion marker yet (touched {age_min:.0f}m ago)"

    return FAILED, node, _reason(text)


def _reason(text: str) -> str:
    """Last line mentioning a known error marker, else a generic message."""
    for line in reversed(text.splitlines()):
        if any(marker in line for marker in ERROR_MARKERS):
            line = line.strip()
            return line[:120] + ("..." if len(line) > 120 else "")
    return "died with no completion marker and no recognized error"


def scan(pattern: str, expect: int = None, running_window_min: float = 30.0):
    """Returns (results, missing_ids).

    results maps task_id (or path, when unparseable) -> (status, node, reason).
    missing_ids lists array indices in range(expect) that produced no log at
    all -- tasks that never started are invisible to any log-scanning approach,
    so they have to be inferred from the expected array size.
    """
    paths = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    results = {}
    for path in paths:
        key = task_id(path)
        results[key if key is not None else path] = classify(path, running_window_min)

    missing_ids = []
    if expect is not None:
        seen = {k for k in results if isinstance(k, int)}
        missing_ids = [i for i in range(expect) if i not in seen]

    return results, missing_ids
