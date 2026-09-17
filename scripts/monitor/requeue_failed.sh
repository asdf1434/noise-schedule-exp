#!/bin/bash

# ==========================================
# Requeue only the array tasks that failed in a previous Slurm array job,
# instead of manually rereading logs (check_slurm_logs.py) and retyping
# --array indices by hand. Works for any array job -- training or eval --
# since the (train_dist, seed) / shard mapping lives inside the .sh script
# itself, keyed off $SLURM_ARRAY_TASK_ID.
#
# Usage: scripts/monitor/requeue_failed.sh <log_pattern> <experiment> [stage] [array_size]
# Example:
#   scripts/monitor/requeue_failed.sh "slurm_exp1_train_1097429_*.out" exp1 train 120
#
# <experiment> is a name from scripts/slurm/experiments/ (see submit.sh --list)
# and <stage> is train (the default) or eval. The stage script needs EXPERIMENT
# in its environment, so this passes it the same way submit.sh does.
#
# A path to a .sh file is accepted in that position too, and is resubmitted as
# is with no EXPERIMENT set. That is how the pre-consolidation launchers were
# requeued, and it still works for a job submitted from one of them:
#   scripts/monitor/requeue_failed.sh "slurm_exp1_pilot_1097429_*.out" \
#       scripts/slurm/delete_later/run_exp1.sh 120
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
EXPERIMENT=$2
STAGE=${3:-train}
ARRAY_SIZE=$4

if [ -z "$LOG_PATTERN" ] || [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <log_pattern> <experiment> [stage] [array_size]" >&2
    echo "Example: $0 \"slurm_exp1_train_1097429_*.out\" exp1 train 120" >&2
    exit 1
fi

# Second argument is either a script path (the old form) or an experiment name.
if [ "${EXPERIMENT%.sh}" != "$EXPERIMENT" ]; then
    SCRIPT=$EXPERIMENT
    EXPERIMENT=""
    ARRAY_SIZE=${3:-}          # old form took array_size here, with no stage
    if [ ! -f "$SCRIPT" ]; then
        echo "No such script: $SCRIPT" >&2
        exit 1
    fi
else
    case "$STAGE" in
        train|eval) SCRIPT="scripts/slurm/$STAGE.sh" ;;
        *) echo "Stage must be train or eval (got '$STAGE')." >&2; exit 1 ;;
    esac
    if [ ! -f "scripts/slurm/experiments/$EXPERIMENT.conf" ]; then
        echo "Unknown experiment '$EXPERIMENT' -- see scripts/slurm/submit.sh --list" >&2
        exit 1
    fi
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
if [ -z "$EXPERIMENT" ]; then
    echo "Requeuing: sbatch --array=$FAILED_IDS $SCRIPT"
    sbatch --array="$FAILED_IDS" "$SCRIPT"
else
    echo "Requeuing: sbatch --array=$FAILED_IDS --export=ALL,EXPERIMENT=$EXPERIMENT $SCRIPT"
    sbatch --array="$FAILED_IDS" \
           --export=ALL,EXPERIMENT="$EXPERIMENT" \
           --job-name="${EXPERIMENT}_${STAGE}" \
           --output="logs/slurm/slurm_${EXPERIMENT}_${STAGE}_%A_%a.out" \
           "$SCRIPT"
fi
