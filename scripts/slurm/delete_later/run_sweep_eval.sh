#!/bin/bash

# ==========================================
# Stage 2 of 3 -- score the training-distribution centre sweep.
#
# Supersedes run_mnist_x10_eval.sh. evaluate_fid.py filters eval_runs/ by
# dataset, so this is a (dataset x shard) array:
#
#   dataset_idx = ID / NUM_SHARDS,  shard = ID % NUM_SHARDS
#
# 2 datasets x 16 shards = 32 tasks. Includes plain mnist, which the shift eval Anything already present in
# scripts do not cover. Anything already scored is skipped.
#
# Sharding is over the FULL folder list (fixed 2026-09-02), so a rerun of shard
# k picks up exactly what shard k left unfinished -- reruns no longer leave gaps.
#
#   PREP=$(sbatch --parsable scripts/slurm/run_sweep_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_sweep_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=sweep_eval
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --array=0-47
#SBATCH --output=logs/slurm/slurm_sweep_eval_%A_%a.out

set -e
mkdir -p logs/slurm
source venv/bin/activate

# Keep in sync with run_sweep_eval_prep.sh.
DATASETS=(mnist mnist_x10 mnist_x0.1)
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
DATASET=${DATASETS[$(( IDX / NUM_SHARDS ))]}
SHARD=$(( IDX % NUM_SHARDS ))

echo "=== task $IDX: dataset=$DATASET shard $SHARD / $NUM_SHARDS ==="
python -u evaluate_fid.py --shard "$SHARD" --num_shards "$NUM_SHARDS" --dataset "$DATASET"
echo "=== task $IDX complete ==="
