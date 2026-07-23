#!/bin/bash

# ==========================================
# Step 1 of 3 for offline FID evaluation on eurosat64 -- caches the real-image
# Inception stats ONCE (needed by every downstream FID computation), scoped
# to the eurosat64 reference set (data/real_eurosat64 / stats name
# "eurosat64_real", kept separate from mnist/eurosat). Requires
# `python -m src.generate_real_samples --dataset eurosat64 --num_samples 5000`
# to have been run once already (not part of this job).
#
# Run this AFTER all run_eurosat64.sh array tasks have finished successfully,
# then chain the rest with dependencies:
#
#   sbatch scripts/slurm/run_eurosat64_eval.sh
#   # note the job id it prints, e.g. "Submitted batch job 1100500"
#   sbatch --dependency=afterok:1100500 scripts/slurm/run_eurosat64_eval_array.sh
#   # note THAT job id too, e.g. "Submitted batch job 1100501"
#   sbatch --dependency=afterok:1100501 scripts/slurm/run_eurosat64_eval_merge.sh
# ==========================================
#SBATCH --job-name=eurosat64_eval_stats
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# improbablex002 segfaults inside torch/clean-fid's CUDA init (see
# run_eurosat64.sh's header for the other known-bad nodes).
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/slurm_eurosat64_eval_stats_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

echo "========================================"
echo "Caching real-image FID stats for eurosat64 (one-time, needed before the eval array)"
echo "========================================"

python -u cache_real_stats.py --dataset eurosat64

echo -e "\n========================================"
echo "Stats cached. Next: sbatch --dependency=afterok:$SLURM_JOB_ID scripts/slurm/run_eurosat64_eval_array.sh"
echo "========================================"
