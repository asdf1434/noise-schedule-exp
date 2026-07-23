#!/bin/bash

# ==========================================
# Step 3 of 3 -- merges the per-shard master_fid_results_shard*_cifar10.json
# files (written by run_cifar10_eval_array.sh's 64 array tasks) into
# results/master_fid_results.json, same merge_fid_shards.py used by every
# other dataset (it globs all shard*.json files regardless of dataset and
# deep-merges them; safe to run after/alongside other datasets' merges since
# experiment names are always prefixed by dataset via src/naming.py).
#
# Run AFTER all 64 run_cifar10_eval_array.sh tasks have finished:
#   sbatch --dependency=afterok:<ARRAY_JOBID> scripts/slurm/run_cifar10_eval_merge.sh
# ==========================================
#SBATCH --job-name=cifar10_eval_merge
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/slurm_cifar10_eval_merge_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Merging shard results -> master_fid_results.json"
echo "========================================"

python -u merge_fid_shards.py

echo -e "\n========================================"
echo "Merge done. master_fid_results.json now also has 120 ds-cifar10__ entries"
echo "(6 dists x 20 seeds), each scored against all 4 sampling schedules per eval epoch."
echo ""
echo "Next: python scripts/plots/aggregate_fid.py  (mean/std across seeds + plotting)"
echo "========================================"
