#!/bin/bash

# ==========================================
# Step 3 of 3 -- merges the per-shard master_fid_results_shard*.json files
# (written by run_exp1_eval_array.sh's 8 array tasks) into a single
# master_fid_results.json.
#
# Run AFTER all 8 run_exp1_eval_array.sh tasks have finished:
#   sbatch --dependency=afterok:<ARRAY_JOBID> scripts/slurm/run_exp1_eval_merge.sh
#
# <ARRAY_JOBID> is the number sbatch printed when submitting
# run_exp1_eval_array.sh (the array job id, not a per-task id like
# <id>_3). afterok waits for every task in that array to exit 0.
# ==========================================
#SBATCH --job-name=exp1_pilot_eval_merge
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/slurm_exp1_pilot_eval_merge_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Merging shard results -> master_fid_results.json"
echo "========================================"

python -u merge_fid_shards.py

echo -e "\n========================================"
echo "Merge done. master_fid_results.json now has 10 experiment entries:"
echo "  uniform_seed0 .. uniform_seed4"
echo "  logit_normal_mu_0.0_sigma_1.0_seed0 .. _seed4"
echo "each scored against all 4 sampling schedules per eval epoch."
echo ""
echo "Next: python aggregate_fid.py  (mean/std across seeds + plotting)"
echo "========================================"
