#!/bin/bash

# ==========================================
# Does giving the model more help shrink the sampling schedule's benefit?
#
# The July result was that inpaint conditioning and the coarse step spacing seem
# to "buy the same thing": on Fashion-MNIST the coarse spacing gained LEAST from
# adding conditioning (-36%) while the fine spacing gained most (-56%), as if
# both were fixing the same problem and the second one you add is worth less.
#
# If that's real, it should be a smooth trade rather than a coincidence: the
# more help the model gets, the less the step spacing should matter. You can't
# see that by turning conditioning on and off -- you have to vary how MUCH it
# gives, which is what this sweep does.
#
#   class                     just the digit label -- tiny hint, but a strong one
#   inpaint known_fraction    0.25 / 0.75 of the image handed over
#                             (0.5 already exists from the earlier sweep)
#   lowres factor             2 / 4 / 7 -- a blocky preview at 14x14, 7x7, 4x4
#
# Prediction being tested: the coarse spacing's advantage shrinks monotonically
# as the amount of help goes up.
#
# Only 3 training distributions, not the usual 6 -- see run_fashion_sizes.sh.
#
# Job array: 180 tasks = 6 variants x 3 distributions x 10 seeds.
#   variant_idx = ID / 30,  dist_idx = (ID % 30) / 10,  seed = ID % 10
# Cap concurrency if the pool can't grant 180 at once: --array=0-179%20
#
# NEEDS NOTHING FIRST: every variant is plain MNIST, so it reuses the existing
# data/real + cached mnist_real stats. Conditioning is a property of the model,
# not of the images FID scores against.
#
# AFTER: score with the existing mnist chain, which picks these up automatically
# (evaluate_fid.py filters by dataset only, not by conditioning):
#   python scripts/monitor/seed_fid_shards.py --dataset mnist --num_shards 64
#   sbatch scripts/slurm/run_exp1_eval_array.sh
#   sbatch --dependency=afterok:<JOBID> scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=mnist_conditioning
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
# (see schedules_to_test in train.py).
#SBATCH --time=03:00:00
#SBATCH --array=0-179
#SBATCH --output=logs/slurm/slurm_mnist_conditioning_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Map array task ID -> (conditioning variant, train_dist, seed)
#
# The two arrays below are parallel: COND_KINDS[i] goes with COND_SETTINGS[i].
# The settings land in the experiment name (see make_exp_name), so the three
# lowres runs don't overwrite each other the way they would have before.
# ==========================================
NUM_SEEDS=10
NUM_DISTS=3
COND_KINDS=(class inpaint inpaint lowres lowres lowres)
COND_SETTINGS=('{}' '{"known_fraction": 0.25}' '{"known_fraction": 0.75}' '{"factor": 2}' '{"factor": 4}' '{"factor": 7}')
DIST_NAMES=(uniform logit_normal logit_normal_peaked)

IDX=$SLURM_ARRAY_TASK_ID
PER_VARIANT=$((NUM_DISTS * NUM_SEEDS))
VARIANT_IDX=$((IDX / PER_VARIANT))
DIST_IDX=$(((IDX % PER_VARIANT) / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
COND=${COND_KINDS[$VARIANT_IDX]}
COND_PARAMS=${COND_SETTINGS[$VARIANT_IDX]}
DIST_NAME=${DIST_NAMES[$DIST_IDX]}

echo "========================================"
echo "Array task $IDX: cond=$COND params=$COND_PARAMS dist=$DIST_NAME seed=$SEED"
echo "========================================"

case "$DIST_NAME" in
  uniform)
    python -u train.py \
        --dataset mnist \
        --conditioning "$COND" \
        --cond_params "$COND_PARAMS" \
        --train_dist uniform \
        --seed "$SEED"
    ;;
  logit_normal)
    python -u train.py \
        --dataset mnist \
        --conditioning "$COND" \
        --cond_params "$COND_PARAMS" \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  logit_normal_peaked)
    python -u train.py \
        --dataset mnist \
        --conditioning "$COND" \
        --cond_params "$COND_PARAMS" \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 0.3}' \
        --seed "$SEED"
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($COND $COND_PARAMS $DIST_NAME seed $SEED) complete."
echo "========================================"
