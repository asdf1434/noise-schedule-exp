#!/bin/bash

# ==========================================
# EuroSAT counterpart to run_exp1.sh + run_exp2.sh combined: all 6 training
# distributions those two covered for MNIST, now pointed at --dataset
# eurosat instead, all at 20 seeds each (exp1 originally only ran
# uniform/logit_normal at 5 seeds -- bumped to 20 here to match exp2's
# quartet for a consistent seed count across all 6 distributions). See
# src/utils.py's get_eurosat_dataloaders for how EuroSAT is reduced to
# MNIST's 28x28 monochrome format (grayscale, resize 64->32, center-crop
# 28x28).
#
# Before submitting this for the first time, the EuroSAT real-image
# reference set must exist (one-time, not part of this array job):
#   python -m src.generate_real_samples --dataset eurosat --num_samples 5000
#
# Job array: 120 independent tasks (SLURM_ARRAY_TASK_ID 0..119), one per
# (train_dist, seed) combo -- 6 distributions x 20 seeds each -- running in
# PARALLEL across GPUs instead of sequentially in one long job:
#   uniform, logit_normal(mu=0,sigma=1), logit_normal_peaked(sigma=0.3),
#   logit_normal_skew_hi(mu=1.5), logit_normal_skew_lo(mu=-1.5), plateau
# 120 tasks concurrently needs more GPUs than a small pilot -- if the
# lab-free QOS/partition pool can't grant that many at once, add a
# concurrency cap by changing --array=0-119 to e.g. --array=0-119%20.
# ==========================================
#SBATCH --job-name=eurosat_train_dists
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# isola-2080ti-4, gpu19-2.drl, gpu20-2.drl: reliably fail CUDA init (each a
# different cuInit error, e.g. CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE on
# isola-2080ti-4) -- same three nodes already excluded in run_exp2.sh, kept
# here rather than dropping their whole partitions since sibling nodes ran
# fine. improbablex002: segfaults inside torch/clean-fid's CUDA init
# (unrelated subsystem to jax, but still bad hardware/drivers -- excluded
# defensively in case future eval-array jobs land there too).
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-119
#SBATCH --output=logs/slurm/slurm_eurosat_train_dists_%A_%a.out

# Exit immediately if a command exits with a non-zero status
set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Map array task ID -> (train_dist, seed): 6 distributions x 20 seeds each
# dist_idx = ID / 20, seed = ID % 20
# ==========================================
NUM_SEEDS=20
DIST_NAMES=(uniform logit_normal logit_normal_peaked logit_normal_skew_hi logit_normal_skew_lo plateau)

IDX=$SLURM_ARRAY_TASK_ID
DIST_IDX=$((IDX / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
DIST_NAME=${DIST_NAMES[$DIST_IDX]}

echo "========================================"
echo "Array task $IDX: dataset=eurosat dist=$DIST_NAME seed=$SEED"
echo "========================================"

case "$DIST_NAME" in
  uniform)
    python -u train.py --dataset eurosat --train_dist uniform --seed "$SEED"
    ;;
  logit_normal)
    python -u train.py \
        --dataset eurosat \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_peaked)
    python -u train.py \
        --dataset eurosat \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 0.3}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_hi)
    python -u train.py \
        --dataset eurosat \
        --train_dist logit_normal \
        --dist_params '{"mu": 1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_lo)
    python -u train.py \
        --dataset eurosat \
        --train_dist logit_normal \
        --dist_params '{"mu": -1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  plateau)
    python -u train.py \
        --dataset eurosat \
        --train_dist plateau_logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0, "uniform_prob": 0.3}' \
        --seed "$SEED"
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($DIST_NAME seed $SEED) complete."
echo "Once ALL 120 array tasks finish, run evaluate_fid.py --dataset eurosat separately -- see"
echo "scripts/slurm/run_eurosat_eval.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
