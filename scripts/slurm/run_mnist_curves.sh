#!/bin/bash

# ==========================================
# Does making digits blockier or softer change which step spacing wins?
#
# Same digits, same 28x28 size, same model, same everything -- only the balance
# of hard black-and-white vs soft grey changes. This is the control that the
# image-size comparison (run_fashion_sizes.sh) can't give, because changing size
# changes both the picture and how well the model fits it at once.
#
#   g=0.5  soft, smudgy      18.8% mid-grey pixels, spectrum slope -2.79
#   g=1.0  ordinary MNIST      9.1%                              -2.66
#   g=2.5  hard stencils       5.9%                              -2.29
#
# The intermediate 0.7 / 1.5 levels were dropped for cost, so this shows the
# direction and sign of the effect but not whether it moves smoothly.
#
# Every level is rescaled to an identical spread of pixel values, so the
# signal-to-noise ratio the model sees at each timestep is unchanged and only
# the picture differs -- see _gamma_transform in src/datasets.py for why that
# matters and why the scaling is anchored at black.
#
# NOTE g=1.0 is *not* pixel-identical to plain MNIST (it carries the same 0.916
# rescale as the others), so don't expect it to reproduce existing MNIST FID
# exactly -- it's the control arm within this experiment, not a reproduction of
# the old one.
#
# Only 3 training distributions, not the usual 6 -- see run_fashion_sizes.sh for
# the reasoning.
#
# Job array: 90 tasks = 3 levels x 3 distributions x 10 seeds.
#   level_idx = ID / 30,  dist_idx = (ID % 30) / 10,  seed = ID % 10
# Cap concurrency if the pool can't grant 90 at once: --array=0-89%20
#
# NEEDS FIRST: real images + cached FID stats for all three levels --
#   sbatch scripts/slurm/run_overnight_prep.sh
# Each level is a different distribution and is scored against its own reference
# images, so FID is NOT comparable between levels -- only the difference between
# step spacings *within* a level means anything.
#
# AFTER: score with run_overnight_eval.sh.
# ==========================================
#SBATCH --job-name=mnist_curves
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# Same known-bad nodes excluded in run_eurosat.sh/run_eurosat64.sh -- see
# their headers for the full rationale (CUDA init failures / segfaults).
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 3h not 1.5h: every checkpoint now generates 9 sets of images instead of 4
# (see schedules_to_test in train.py).
#SBATCH --time=03:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/slurm/slurm_mnist_curves_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Map array task ID -> (dataset, train_dist, seed)
# ==========================================
NUM_SEEDS=10
NUM_DISTS=3
DATASETS=(mnist_g050 mnist_g100 mnist_g250)
DIST_NAMES=(uniform logit_normal logit_normal_peaked)

IDX=$SLURM_ARRAY_TASK_ID
PER_LEVEL=$((NUM_DISTS * NUM_SEEDS))
LEVEL_IDX=$((IDX / PER_LEVEL))
DIST_IDX=$(((IDX % PER_LEVEL) / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
DATASET=${DATASETS[$LEVEL_IDX]}
DIST_NAME=${DIST_NAMES[$DIST_IDX]}

echo "========================================"
echo "Array task $IDX: dataset=$DATASET dist=$DIST_NAME seed=$SEED"
echo "========================================"

case "$DIST_NAME" in
  uniform)
    python -u train.py --dataset "$DATASET" --train_dist uniform --seed "$SEED"
    ;;
  logit_normal)
    python -u train.py \
        --dataset "$DATASET" \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_peaked)
    python -u train.py \
        --dataset "$DATASET" \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 0.3}' \
        --seed "$SEED"
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($DATASET $DIST_NAME seed $SEED) complete."
echo "========================================"
