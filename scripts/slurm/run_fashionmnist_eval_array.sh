#!/bin/bash

# ==========================================
# Step 2 of 3 -- scores all fashion_mnist eval_runs/ folders in parallel via
# a Slurm job array, the same pattern as run_eurosat_eval_array.sh but
# scoped to dataset=fashion_mnist.
#
# Run AFTER run_fashionmnist_eval.sh (real-image stats must already be cached):
#   sbatch --dependency=afterok:<STATS_JOBID> scripts/slurm/run_fashionmnist_eval_array.sh
#
# Each task writes its own master_fid_results_shard<N>_fashion_mnist.json --
# run run_fashionmnist_eval_merge.sh afterward to combine them into
# master_fid_results.json.
# ==========================================
#SBATCH --job-name=fashionmnist_eval_array
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --array=0-63
#SBATCH --output=logs/slurm/slurm_fashionmnist_eval_array_%A_%a.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

NUM_SHARDS=64

echo "========================================"
echo "Shard $SLURM_ARRAY_TASK_ID / $NUM_SHARDS (dataset=fashion_mnist)"
echo "========================================"

python -u evaluate_fid.py --shard "$SLURM_ARRAY_TASK_ID" --num_shards "$NUM_SHARDS" --dataset fashion_mnist

echo -e "\n========================================"
echo "Shard $SLURM_ARRAY_TASK_ID complete."
echo "Once ALL $NUM_SHARDS array tasks finish, run merge_fid_shards.py -- see"
echo "scripts/slurm/run_fashionmnist_eval_merge.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
