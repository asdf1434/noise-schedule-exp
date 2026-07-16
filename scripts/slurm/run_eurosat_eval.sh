#!/bin/bash

# ==========================================
# Step 1 of 3 for offline FID evaluation on EuroSAT -- caches the real-image
# Inception stats ONCE (needed by every downstream FID computation), scoped
# to the EuroSAT reference set (data/real_eurosat / stats name
# "eurosat_real", kept separate from MNIST's). Requires
# `python -m src.generate_real_samples --dataset eurosat --num_samples 5000`
# to have been run once already (not part of this job).
#
# Run this AFTER all 10 run_eurosat.sh array tasks have finished
# successfully, then chain the rest with dependencies:
#
#   sbatch scripts/slurm/run_eurosat_eval.sh
#   # note the job id it prints, e.g. "Submitted batch job 1100500"
#   sbatch --dependency=afterok:1100500 scripts/slurm/run_eurosat_eval_array.sh
#   # note THAT job id too, e.g. "Submitted batch job 1100501"
#   sbatch --dependency=afterok:1100501 scripts/slurm/run_eurosat_eval_merge.sh
# ==========================================
#SBATCH --job-name=eurosat_pilot_eval_stats
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/slurm_eurosat_pilot_eval_stats_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Caching real-image FID stats for eurosat (one-time, needed before the eval array)"
echo "========================================"

python -u cache_real_stats.py --dataset eurosat

echo -e "\n========================================"
echo "Stats cached. Next: sbatch --dependency=afterok:$SLURM_JOB_ID scripts/slurm/run_eurosat_eval_array.sh"
echo "========================================"
