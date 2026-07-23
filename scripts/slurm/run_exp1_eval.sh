#!/bin/bash

# ==========================================
# Step 1 of 3 for offline FID evaluation -- caches the real-image Inception
# stats ONCE (needed by every downstream FID computation). Run this AFTER
# all 10 run_exp1.sh array tasks have finished successfully, then chain the
# rest with dependencies so you don't have to babysit squeue:
#
#   sbatch scripts/slurm/run_exp1_eval.sh
#   # note the job id it prints, e.g. "Submitted batch job 1100500"
#   sbatch --dependency=afterok:1100500 scripts/slurm/run_exp1_eval_array.sh
#   # note THAT job id too, e.g. "Submitted batch job 1100501"
#   sbatch --dependency=afterok:1100501 scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=exp1_pilot_eval_stats
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/slurm_exp1_pilot_eval_stats_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Caching real-image FID stats (one-time, needed before the eval array)"
echo "========================================"

python -u cache_real_stats.py

echo -e "\n========================================"
echo "Stats cached. Next: sbatch --dependency=afterok:$SLURM_JOB_ID scripts/slurm/run_exp1_eval_array.sh"
echo "========================================"
