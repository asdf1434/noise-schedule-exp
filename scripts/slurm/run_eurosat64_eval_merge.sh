#!/bin/bash

# ==========================================
# Step 3 of 3 -- merges the per-shard master_fid_results_shard*_eurosat64.json
# files (written by run_eurosat64_eval_array.sh's 64 array tasks) into
# results/master_fid_results.json, same merge_fid_shards.py used by the
# mnist/eurosat pipelines (it globs all shard*.json files regardless of
# dataset and deep-merges them; safe to run after/alongside the other merges
# since the datasets' experiment names never collide -- the dataset is part of
# the canonical exp_name, e.g. ds-eurosat64__...).
#
# Run AFTER all 64 run_eurosat64_eval_array.sh tasks have finished:
#   sbatch --dependency=afterok:<ARRAY_JOBID> scripts/slurm/run_eurosat64_eval_merge.sh
#
# <ARRAY_JOBID> is the number sbatch printed when submitting
# run_eurosat64_eval_array.sh (the array job id, not a per-task id like
# <id>_3). afterok waits for every task in that array to exit 0.
# ==========================================
#SBATCH --job-name=eurosat64_eval_merge
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/slurm_eurosat64_eval_merge_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Merging shard results -> master_fid_results.json"
echo "========================================"

python -u merge_fid_shards.py

echo -e "\n========================================"
echo "Merge done. master_fid_results.json now also has the eurosat64 entries"
echo "(ds-eurosat64__cond-none__dist-*__seed-*), each scored against all 4"
echo "sampling schedules per eval epoch."
echo ""
echo "Next: python scripts/plots/aggregate_fid.py  (mean/std across seeds + plotting)"
echo "========================================"
