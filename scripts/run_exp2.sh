#!/bin/bash

# ==========================================
# Experiment 2: are training distributions more distinct from each other
# than uniform vs. logit_normal(mu=0, sigma=1) (exp 1's null result)?
#
# 4 new training distributions x 20 seeds = 80 independent tasks, run in
# PARALLEL across GPUs (same job-array pattern as run_exp1.sh). Each task
# does ONE training run (~42 min at ~25s/epoch x 100 epochs).
#
# Distributions chosen by measured overlap/TV-distance vs. uniform (see
# scratch analysis in conversation): sigma=0.3 is the most distinct
# (TV=0.665, 33.5% overlap); the skewed mu=+-1.5 options are next
# (TV=0.475, 52.5% overlap); plateau mixes uniform + logit_normal(0,1)
# directly. uniform and logit_normal(0,1) themselves are NOT rerun here --
# reuse exp 1's runs (see aggregated_fid_results.json).
# ==========================================
#SBATCH --job-name=exp2_train_dists
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# isola-2080ti-4, gpu19-2.drl, and gpu20-2.drl reliably fail CUDA init
# (cuInit errors, each a different error code) -- excluded here rather than
# dropping their whole partitions, since sibling nodes (isola-2080ti-1/2/3,
# gpu19-1.drl) ran fine.
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-79
#SBATCH --output=logs/slurm/slurm_exp2_train_dists_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Map array task ID -> (train_dist, dist_params, seed)
# 4 distributions x 20 seeds each: dist_idx = ID / 20, seed = ID % 20
# ==========================================
NUM_SEEDS=20

DIST_NAMES=(logit_normal_peaked logit_normal_skew_hi logit_normal_skew_lo plateau)

IDX=$SLURM_ARRAY_TASK_ID
DIST_IDX=$((IDX / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
DIST_NAME=${DIST_NAMES[$DIST_IDX]}

echo "========================================"
echo "Array task $IDX: dist=$DIST_NAME seed=$SEED"
echo "========================================"

case "$DIST_NAME" in
  logit_normal_peaked)
    python -u train.py \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 0.3}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_hi)
    python -u train.py \
        --train_dist logit_normal \
        --dist_params '{"mu": 1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_skew_lo)
    python -u train.py \
        --train_dist logit_normal \
        --dist_params '{"mu": -1.5, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  plateau)
    python -u train.py \
        --train_dist plateau_logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0, "uniform_prob": 0.3}' \
        --seed "$SEED"
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($DIST_NAME seed $SEED) complete."
echo "Once ALL 80 array tasks finish, run evaluate_fid.py separately -- reuse"
echo "run_exp1_eval_array.sh + run_exp1_eval_merge.sh (they walk all of"
echo "eval_runs/ generically, no changes needed for this experiment)."
echo "========================================"
