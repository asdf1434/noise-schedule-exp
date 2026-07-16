#!/bin/bash

# ==========================================
# Submits the full training -> FID pipeline in one shot, chaining all four
# stages with `sbatch --dependency=afterok` so you don't have to babysit
# squeue and manually submit each next stage yourself:
#
#   1. train array          (e.g. run_exp1.sh / run_exp2.sh)
#   2. real-image stats      (run_exp1_eval_stats.sh)
#   3. FID eval array         (run_exp1_eval_array.sh)
#   4. merge shards           (run_exp1_eval_merge.sh)
#
# afterok means a stage only starts once EVERY task of the previous stage
# exits 0. If any task fails, the chain stops there (same checkpoint you'd
# hit babysitting it manually) -- use scripts/monitor/requeue_failed.sh to
# requeue just the failed tasks, then resubmit the remaining stages by hand
# (or rerun this script; already-scored folders/checkpoints are skipped by
# evaluate_fid.py's resume logic, so it's safe to rerun the train stage
# too as long as train.py's own resume/skip behavior covers it).
#
# Usage: scripts/slurm/run_pipeline.sh <train_script.sh>
# Example: scripts/slurm/run_pipeline.sh scripts/slurm/run_exp1.sh
#          scripts/slurm/run_pipeline.sh scripts/slurm/run_exp2.sh
# ==========================================

set -e

TRAIN_SCRIPT=$1

if [ -z "$TRAIN_SCRIPT" ]; then
    echo "Usage: $0 <train_script.sh>" >&2
    echo "Example: $0 scripts/slurm/run_exp1.sh" >&2
    exit 1
fi

mkdir -p logs/slurm

TRAIN_JOBID=$(sbatch --parsable "$TRAIN_SCRIPT")
echo "1/4 train array:  job $TRAIN_JOBID  ($TRAIN_SCRIPT)"

STATS_JOBID=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOBID" scripts/slurm/run_exp1_eval.sh)
echo "2/4 real stats:    job $STATS_JOBID  (after $TRAIN_JOBID)"

EVAL_JOBID=$(sbatch --parsable --dependency=afterok:"$STATS_JOBID" scripts/slurm/run_exp1_eval_array.sh)
echo "3/4 FID eval array: job $EVAL_JOBID  (after $STATS_JOBID)"

MERGE_JOBID=$(sbatch --parsable --dependency=afterok:"$EVAL_JOBID" scripts/slurm/run_exp1_eval_merge.sh)
echo "4/4 merge:         job $MERGE_JOBID  (after $EVAL_JOBID)"

echo ""
echo "Submitted. Track with: squeue -u \$USER"
echo "If a stage fails, check logs first:"
echo "  python scripts/monitor/check_slurm_logs.py \"*${TRAIN_JOBID}*\""
echo "Then requeue just the failed tasks, e.g.:"
echo "  scripts/monitor/requeue_failed.sh \"*${TRAIN_JOBID}_*.out\" $TRAIN_SCRIPT"
