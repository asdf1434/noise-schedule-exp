#!/bin/bash

# ==========================================
# Step 1 of 3 -- reference images + cached FID stats for the new datasets.
#
# Every dataset is scored against its OWN folder of real images, so a new
# dataset can't be evaluated until this has run. The two experiments launching
# tonight need five new ones:
#
#   fashion_mnist_r14 / _r56    different image size -> different real images
#   mnist_g050 / g100 / g250    different tone curve -> different real images
#
# Two datasets deliberately need nothing here:
#   fashion_mnist_r28  same pixels as fashion_mnist, points at its real_dir
#   mnist (conditioning) conditioning changes the model, not the images
#
# Scoring g050 samples against ordinary MNIST's reference images would measure
# the tone-curve difference rather than sample quality, which is why each level
# gets its own set.
#
# Runs sequentially in one job -- it's minutes of work, not worth an array.
# Safe to re-run: generate_real_samples overwrites, and cache_real_stats skips
# anything already cached.
#
#   sbatch scripts/slurm/run_overnight_prep.sh
#
# THEN (needs the reference images to exist):
#   python -m scripts.plots.plot_spectral_slope
# to record the predictions before any results land.
# ==========================================
#SBATCH --job-name=overnight_prep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm_overnight_prep_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

DATASETS=(fashion_mnist_r14 fashion_mnist_r56 mnist_g050 mnist_g100 mnist_g250)

for DS in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Preparing $DS"
  echo "========================================"
  python -u -m src.generate_real_samples --dataset "$DS"
  python -u cache_real_stats.py --dataset "$DS"
done

echo -e "\n========================================"
echo "Prep complete for: ${DATASETS[*]}"
echo "Next: python -m scripts.plots.plot_spectral_slope   (record predictions)"
echo "Then submit the training arrays."
echo "========================================"
