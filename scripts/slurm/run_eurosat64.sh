#!/bin/bash

# ==========================================
# eurosat64 counterpart to run_eurosat.sh: the same 6 training distributions
# x 20 seeds, but at EuroSAT's NATIVE 64x64 resolution (--dataset eurosat64)
# instead of the 28x28 MNIST-matched crop. Image geometry is read from the
# dataset registry in src/datasets.py, so nothing here is hardcoded to 28x28.
#
# NOTE ON COST/TIME: 64x64 is ~5x the pixels of 28x28, so each epoch trains
# noticeably slower -- the train --time below is bumped from run_eurosat.sh's
# 01:30:00 to 06:00:00. If your smoke test showed a different per-epoch time,
# adjust --time (and/or --epochs on the train.py calls) accordingly. To run a
# smaller/cheaper pilot, drop NUM_SEEDS or narrow --array (e.g. --array=0-5
# for one seed of each of the 6 dists).
#
# Before submitting this for the first time, the eurosat64 real-image
# reference set + cached FID stats must exist (one-time, not part of this job):
#   python -m src.generate_real_samples --dataset eurosat64 --num_samples 5000
#   python cache_real_stats.py --dataset eurosat64     # (run_eurosat64_eval.sh does this)
#
# Job array: 120 independent tasks (SLURM_ARRAY_TASK_ID 0..119), one per
# (train_dist, seed) combo -- 6 distributions x 20 seeds each -- running in
# PARALLEL across GPUs:
#   uniform, logit_normal(mu=0,sigma=1), logit_normal_peaked(sigma=0.3),
#   logit_normal_skew_hi(mu=1.5), logit_normal_skew_lo(mu=-1.5), plateau
# 120 tasks concurrently needs more GPUs than a small pilot -- if the
# lab-free QOS/partition pool can't grant that many at once, add a
# concurrency cap by changing --array=0-119 to e.g. --array=0-119%20.
# ==========================================
#SBATCH --job-name=eurosat64_train_dists
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# isola-2080ti-4, gpu19-2.drl, gpu20-2.drl: reliably fail CUDA init. improbablex002
# segfaults inside torch/clean-fid's CUDA init. Same known-bad nodes excluded
# in run_eurosat.sh -- see its header for the full rationale.
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/slurm/slurm_eurosat64_train_dists_%A_%a.out

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
echo "Array task $IDX: dataset=eurosat64 dist=$DIST_NAME seed=$SEED"
echo "========================================"

case "$DIST_NAME" in
  uniform)
    python -u train.py --dataset eurosat64 --train_dist uniform --seed "$SEED"
    ;;
  logit_normal)
    python -u train.py \
        --dataset eurosat64 \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_peaked)
    python -u train.py \
        --dataset eurosat64 \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 0.3}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_hi)
    python -u train.py \
        --dataset eurosat64 \
        --train_dist logit_normal \
        --dist_params '{"mu": 1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_lo)
    python -u train.py \
        --dataset eurosat64 \
        --train_dist logit_normal \
        --dist_params '{"mu": -1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  plateau)
    python -u train.py \
        --dataset eurosat64 \
        --train_dist plateau_logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0, "uniform_prob": 0.3}' \
        --seed "$SEED"
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($DIST_NAME seed $SEED) complete."
echo "Once ALL 120 array tasks finish, run evaluate_fid.py --dataset eurosat64 separately -- see"
echo "scripts/slurm/run_eurosat64_eval.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
