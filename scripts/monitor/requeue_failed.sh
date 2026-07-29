#!/bin/bash

# ==========================================
# Requeue only the array tasks that failed in a previous Slurm array job,
# instead of manually rereading logs (check_slurm_logs.py) and retyping
# --array indices by hand. Works for any array job -- training or eval --
# since the (train_dist, seed) / shard mapping lives inside the .sh script
# itself, keyed off $SLURM_ARRAY_TASK_ID.
#
# Usage: scripts/monitor/requeue_failed.sh <log_pattern> <original_script.sh> [array_size]
# Example:
#   scripts/monitor/requeue_failed.sh "slurm_exp1_pilot_1097429_*.out" scripts/slurm/run_exp1.sh 120
#
# <log_pattern> is relative to logs/slurm/ and should match only the array
# job you want to check (include the job ID, e.g. slurm_exp1_pilot_1097429_*.out,
# not slurm_exp1_pilot_*.out, or you'll pick up failures from unrelated runs).
#
# Pass [array_size] (the --array width of the original job, e.g. 120) to also
# requeue tasks that never started: those write no log at all, so they can only
# be found by diffing against the expected index range.
# ==========================================

set -e

LOG_PATTERN=$1
SCRIPT=$2
ARRAY_SIZE=$3

if [ -z "$LOG_PATTERN" ] || [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <log_pattern> <original_script.sh> [array_size]" >&2
    echo "Example: $0 \"slurm_exp1_pilot_1097429_*.out\" scripts/slurm/run_exp1.sh 120" >&2
    exit 1
fi

if [ -n "$ARRAY_SIZE" ]; then
    FAILED_IDS=$(python scripts/monitor/list_failed_tasks.py "$LOG_PATTERN" --expect "$ARRAY_SIZE")
else
    FAILED_IDS=$(python scripts/monitor/list_failed_tasks.py "$LOG_PATTERN")
fi

if [ -z "$FAILED_IDS" ]; then
    echo "No failed tasks found matching logs/slurm/$LOG_PATTERN"
    exit 0
fi

echo "Failed array tasks: $FAILED_IDS"
echo "Requeuing: sbatch --array=$FAILED_IDS $SCRIPT"
sbatch --array="$FAILED_IDS" "$SCRIPT"
