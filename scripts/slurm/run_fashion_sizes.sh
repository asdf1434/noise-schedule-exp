#!/bin/bash

# ==========================================
# Does image size change which step spacing wins?
#
# Fashion-MNIST trained at three sizes with everything else held fixed, so the
# only thing varying is the picture itself. The two directions are deliberately
# not symmetric:
#
#   14  a real downsample -- fine detail is genuinely gone
#   28  native, untouched
#   56  interpolated up from 28 -- more pixels, but no more detail
#
# If the best step spacing moves at 14 but not at 56, that's about what's in the
# image. If it moves at 56 too, it's about pixel count and how much of the image
# the network sees at once, which would be a very different explanation.
#
# Only 3 training distributions, not the usual 6: earlier sweeps found uniform /
# logit_normal / plateau statistically indistinguishable and the two skewed ones
# consistently-but-boringly worse, so this keeps two of the interchangeable ones
# (to confirm they still agree at each size) plus the too-narrow sigma=0.3 one,
# which is where an interaction with step spacing has actually shown up before.
#
# Job array: 90 tasks = 3 sizes x 3 distributions x 10 seeds.
#   size_idx = ID / 30,  dist_idx = (ID % 30) / 10,  seed = ID % 10
# If the lab-free pool can't grant 90 GPUs at once, cap it:
#   --array=0-89%20
#
# NEEDS FIRST: real images + cached FID stats for the two new sizes --
#   sbatch scripts/slurm/run_overnight_prep.sh
# (the 28px arm reuses fashion_mnist's existing real images and stats.)
#
# AFTER: score with run_overnight_eval.sh, which covers all three sizes.
# ==========================================
#SBATCH --job-name=fashion_sizes
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# Same known-bad nodes excluded in run_eurosat.sh/run_eurosat64.sh -- see
# their headers for the full rationale (CUDA init failures / segfaults).
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 3h not 1.5h: every checkpoint now generates 9 sets of images instead of 4
# (see schedules_to_test in train.py). The 56x56 runs are the slowest here.
#SBATCH --time=03:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/slurm/slurm_fashion_sizes_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Map array task ID -> (dataset, train_dist, seed)
# ==========================================
NUM_SEEDS=10
NUM_DISTS=3
# r28 is the same data as plain "fashion_mnist" under a separate name, so this
# sweep gets fresh experiment names instead of overwriting the earlier one --
# see the DATASETS entry in src/datasets.py for why that matters.
DATASETS=(fashion_mnist_r14 fashion_mnist_r28 fashion_mnist_r56)
DIST_NAMES=(uniform logit_normal logit_normal_peaked)

IDX=$SLURM_ARRAY_TASK_ID
PER_SIZE=$((NUM_DISTS * NUM_SEEDS))
SIZE_IDX=$((IDX / PER_SIZE))
DIST_IDX=$(((IDX % PER_SIZE) / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
DATASET=${DATASETS[$SIZE_IDX]}
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
