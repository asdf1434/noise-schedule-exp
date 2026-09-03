#!/bin/bash

# ==========================================
# Stage 2 of 3 -- score the mnist_x10 eval_runs/ folders.
#
# Only the 18 tasks of job 1659473 that survived completed training (the other
# 22 died on V100 nodes), so this scores prior seeds 0-9, tuned seeds 0-5, and
# c1.5 seeds 8-9: 18 experiments x 10 epochs x 9 step spacings = 1,620 folders,
# ~101 per shard. Anything already scored is skipped, provided stage 1 seeded
# the shard files.
#
# The point of scoring this partial set before requeueing the rest is that
# prior-vs-tuned is the experiment's load-bearing assumption: if the misplaced
# prior does NOT score worse than the hand-tuned mu, the shift has no dynamic
# range and neither InfoNoise arm can demonstrate anything. prior has all 10
# seeds and tuned has 6, which is enough to see that gap.
#
# MUST run after run_mnist_x10_eval_prep.sh.
#
#   PREP=$(sbatch --parsable scripts/slurm/run_mnist_x10_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_mnist_x10_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
#
# afterok requires EVERY task to exit 0. If any fail, the chain stalls at
# DependencyNeverSatisfied -- requeue with scripts/monitor/requeue_failed.sh,
# then retarget the stalled merge:
#   scontrol update jobid=<MERGE_JOBID> dependency=afterok:<REQUEUE_JOBID>
# ==========================================
#SBATCH --job-name=mnist_x10_eval
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --array=0-15
#SBATCH --output=logs/slurm/slurm_mnist_x10_eval_%A_%a.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

# Keep in sync with run_mnist_x10_eval_prep.sh.
DATASET=mnist_x10
NUM_SHARDS=16

EXPECTED_MAX=$(( NUM_SHARDS - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds $NUM_SHARDS shards." \
         "Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

SHARD=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Task $SHARD: dataset=$DATASET shard $SHARD / $NUM_SHARDS"
echo "========================================"

python -u evaluate_fid.py --shard "$SHARD" --num_shards "$NUM_SHARDS" --dataset "$DATASET"

echo -e "\n========================================"
echo "Shard $SHARD complete. Merge with scripts/slurm/run_exp1_eval_merge.sh"
echo "========================================"
