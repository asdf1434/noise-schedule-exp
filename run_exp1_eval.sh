#!/bin/bash

# ==========================================
# Offline FID evaluation -- run this AFTER all 10 run_exp1.sh array tasks
# have finished successfully. Submit it with a dependency so it waits
# automatically instead of you having to babysit squeue:
#
#   sbatch --dependency=afterok:<ARRAY_JOBID> run_exp1_eval.sh
#
# <ARRAY_JOBID> is the number sbatch printed when you submitted run_exp1.sh
# (e.g. "Submitted batch job 1097500" -> use 1097500, NOT the per-task IDs
# like 1097500_3). afterok waits for every task in that array to exit 0.
# ==========================================
#SBATCH --job-name=exp1_pilot_eval
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/slurm_exp1_pilot_eval_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Computing FID across all eval_runs/ -> master_fid_results.json"
echo "========================================"

python -u evaluate_fid.py

echo -e "\n========================================"
echo "Pilot done. master_fid_results.json now has 10 experiment entries:"
echo "  uniform_seed0 .. uniform_seed4"
echo "  logit_normal_mu_0.0_sigma_1.0_seed0 .. _seed4"
echo "each scored against all 4 sampling schedules per eval epoch."
echo ""
echo "NOTE: plot.py only knows how to plot a single named experiment -- it does"
echo "NOT yet aggregate mean/std across the 5 seeds per (train_dist, schedule)"
echo "cell. That aggregation + comparison step still needs to be written before"
echo "drawing conclusions from this data."
echo "========================================"
