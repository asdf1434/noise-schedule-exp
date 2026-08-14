#!/bin/bash

# ==========================================
# Q2 conditioning sweep: eurosat64 with --conditioning inpaint, the same 6
# training distributions x 20 seeds used for this dataset's cond-none sweep.
# EuroSAT64 trains at native 64x64 -- ~5x the pixels of 28x28, so time budget matches run_eurosat64.sh's bump to 06:00:00.
#
# Does NOT need any new real-image reference set or cached FID stats --
# those are dataset-scoped (already produced by the cond-none phase's
# generate_real_samples.py / cache_real_stats.py runs) and conditioning's
# eval_ref_images come straight from the training dataloader itself
# (train.py withholds up to 1000 images from training for this).
#
# Job array: 120 independent tasks (SLURM_ARRAY_TASK_ID 0..119), one per
# (train_dist, seed) combo -- 6 distributions x 20 seeds each -- running in
# PARALLEL across GPUs:
#   uniform, logit_normal(mu=0,sigma=1), logit_normal_peaked(sigma=0.3),
#   logit_normal_skew_hi(mu=1.5), logit_normal_skew_lo(mu=-1.5), plateau
# 120 tasks concurrently needs more GPUs than a small pilot -- if the
# lab-free QOS/partition pool can't grant that many at once, add a
# concurrency cap by changing --array=0-119 to e.g. --array=0-119%20.
#
# After ALL 120 tasks finish, no new eval pipeline is needed either --
# the existing run_eurosat64_eval*.sh chain already scores every
# experiment under this dataset regardless of conditioning (evaluate_fid.py
# filters by dataset only). Just re-run:
#   sbatch scripts/slurm/run_eurosat64_eval_array.sh   (after stats are cached)
#   sbatch --dependency=afterok:<ARRAY_JOBID> scripts/slurm/run_eurosat64_eval_merge.sh
# ==========================================
#SBATCH --job-name=eurosat64_inpaint
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/slurm/slurm_eurosat64_inpaint_%A_%a.out

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
echo "Array task $IDX: dataset=eurosat64 conditioning=inpaint dist=$DIST_NAME seed=$SEED"
echo "========================================"

case "$DIST_NAME" in
  uniform)
    python -u train.py --dataset eurosat64 --conditioning inpaint --train_dist uniform --seed "$SEED"
    ;;
  logit_normal)
    python -u train.py \
        --dataset eurosat64 \
        --conditioning inpaint \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_peaked)
    python -u train.py \
        --dataset eurosat64 \
        --conditioning inpaint \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 0.3}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_hi)
    python -u train.py \
        --dataset eurosat64 \
        --conditioning inpaint \
        --train_dist logit_normal \
        --dist_params '{"mu": 1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_lo)
    python -u train.py \
        --dataset eurosat64 \
        --conditioning inpaint \
        --train_dist logit_normal \
        --dist_params '{"mu": -1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  plateau)
    python -u train.py \
        --dataset eurosat64 \
        --conditioning inpaint \
        --train_dist plateau_logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0, "uniform_prob": 0.3}' \
        --seed "$SEED"
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($DIST_NAME seed $SEED) complete."
echo "Once ALL 120 array tasks finish, re-run the existing eval chain for eurosat64"
echo "(run_eurosat64_eval_array.sh -> run_eurosat64_eval_merge.sh) to pick up these"
echo "cond-inpaint runs alongside the cond-none ones already scored."
echo "========================================"
