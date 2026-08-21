#!/bin/bash

# ==========================================
# Stage 2 of 3 -- score every eval_runs/ folder for the four grid datasets.
#
# evaluate_fid.py filters eval_runs/ by dataset (each is scored against its own
# real images), so this is a (dataset x shard) array rather than a plain shard
# array -- the same shape as run_overnight_eval.sh:
#
#   dataset_idx = ID / NUM_SHARDS,  shard = ID % NUM_SHARDS
#
# 4 datasets x 16 shards = 64 tasks. Shard files are named per (shard, dataset),
# so all 64 tasks write to different files and never race.
#
# Work per dataset: the grid adds 3 conditioning variants x 5 seeds = 15 runs,
# each with 10 eval epochs x 9 step spacings = 1,350 new folders, so ~85 per
# shard. Everything already scored is skipped, provided the prep stage seeded
# the shard files -- which is the whole reason that stage exists.
#
# MUST run after run_infonoise_grid_eval_prep.sh. Without the cached real-image
# stats, evaluate_fid.py exits telling you to run cache_real_stats.py; without
# the shard seeding, the merge afterwards drops previously scored cells.
#
#   PREP=$(sbatch --parsable scripts/slurm/run_infonoise_grid_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_infonoise_grid_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
#
# afterok requires EVERY task to exit 0. If any fail (bad node, preemption),
# the chain stalls at DependencyNeverSatisfied and stays pending forever --
# requeue with scripts/monitor/requeue_failed.sh, then either resubmit the
# merge by hand or retarget the stalled job:
#   scontrol update jobid=<MERGE_JOBID> dependency=afterok:<REQUEUE_JOBID>
# ==========================================
#SBATCH --job-name=infonoise_grid_eval
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --array=0-63
#SBATCH --output=logs/slurm/slurm_infonoise_grid_eval_%A_%a.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

# Keep in sync with run_infonoise_grid_eval_prep.sh.
DATASETS=(mnist fashion_mnist cifar10 eurosat64)
NUM_SHARDS=16

TOTAL=$(( ${#DATASETS[@]} * NUM_SHARDS ))
EXPECTED_MAX=$(( TOTAL - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds the grid" \
         "(${#DATASETS[@]} datasets x $NUM_SHARDS shards = $TOTAL)." \
         "Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
DS_IDX=$(( IDX / NUM_SHARDS ))
SHARD=$(( IDX % NUM_SHARDS ))
DATASET=${DATASETS[$DS_IDX]}

echo "========================================"
echo "Task $IDX: dataset=$DATASET shard $SHARD / $NUM_SHARDS"
echo "========================================"

python -u evaluate_fid.py --shard "$SHARD" --num_shards "$NUM_SHARDS" --dataset "$DATASET"

echo -e "\n========================================"
echo "Task $IDX ($DATASET shard $SHARD) complete."
echo "Once ALL $TOTAL tasks finish, merge with scripts/slurm/run_exp1_eval_merge.sh"
echo "(submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
