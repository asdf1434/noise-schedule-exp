#!/bin/bash

# ==========================================
# InfoNoise generalization grid, fast datasets (28x28 / 32x32).
#
# The question is NOT "what is the best gate pivot" -- run_infonoise.sh already
# sweeps that on plain MNIST. The question here is whether an online,
# information-guided training noise distribution helps ACROSS conditions, or
# only in the one easy setting it was first tried in.
#
# That matters because this project's own results say the schedule axis is
# dataset-dependent: the sampling-schedule benefit wins on mnist and
# fashion_mnist, goes non-significant on cifar10, and REVERSES on eurosat64.
# A method evaluated only on mnist is evaluated at the easy end of that
# gradient. Conditioning is the second axis: lowres and inpaint each hand the
# model part of the answer, so less of the image is left to invent and the
# remaining uncertainty sits somewhere else along the path -- exactly the kind
# of shift that should move where a good schedule spends its batches.
# (`class` is deliberately left out for now.)
#
#   3 datasets x 3 conditioning variants x 5 seeds = 45 tasks (~42 min each)
#
# eurosat64 is NOT here -- it trains at native 64x64 and needs a 12h walltime
# instead of 3h. It gets run_infonoise_grid64.sh, same grid shape. Submit both.
#
# BASELINES -- ALREADY ON DISK, NO GPU TIME SPENT. The control is
# logit_normal(mu=0, sigma=1), the prior InfoNoise warms up from. Those runs
# already exist at 20 seeds for EVERY cell in this grid (see run_fashionmnist.sh,
# run_cifar10.sh, run_*_lowres.sh, run_*_inpaint.sh), all with default
# --cond_params, so the experiment names line up exactly and every cell has a
# paired control. Dropping `class` is what makes that true -- it was only ever
# run on mnist, so it was the one variant with missing controls.
#
# To add InfoNoise-vs-control runs in one job instead of reusing exp results,
# set ARMS=(infonoise baseline) below and widen --array -- the guard in the
# body prints the right value.
#
# GATE PIVOT. Set GATE_C from whatever run_infonoise.sh's FID says wins. Until
# then it stays on the paper's auto rule. Whatever you pick is held FIXED
# across every cell -- letting c vary per dataset would confound "InfoNoise
# generalizes" with "c was retuned per dataset", which is the thing the paper
# claims not to need.
#
# Prerequisites -- each dataset scores against its OWN real images, so verify
# before submitting or the FID stage silently has nothing to compare against:
#   for d in mnist fashion_mnist cifar10; do ls -d data/real*/ | grep -i $d; done
#
# Submit:
#   sbatch scripts/slurm/run_infonoise_grid.sh
#   sbatch scripts/slurm/run_infonoise_grid64.sh
# Score both afterwards with run_infonoise_grid_eval.sh (all four datasets).
# ==========================================
#SBATCH --job-name=infonoise_grid
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/slurm/slurm_infonoise_grid_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Grid. Task ID decomposes as:
#   dataset_idx = ID / (CONDS * SEEDS),  cond_idx = (ID / SEEDS) % CONDS,
#   seed        = ID % SEEDS
# ==========================================
DATASETS=(mnist fashion_mnist cifar10)
CONDS=(none lowres inpaint)
ARMS=(infonoise)
NUM_SEEDS=5

# Empty --dist_params keeps the auto gate rule and the name "dist-infonoise".
# Set e.g. GATE_C='{"gate_c": 0.15}' once the pilot has been scored.
GATE_C='{}'

TOTAL=$(( ${#DATASETS[@]} * ${#CONDS[@]} * ${#ARMS[@]} * NUM_SEEDS ))
EXPECTED_MAX=$(( TOTAL - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds the grid" \
         "(${#DATASETS[@]} datasets x ${#CONDS[@]} conds x ${#ARMS[@]} arms" \
         "x $NUM_SEEDS seeds = $TOTAL)." \
         "Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
PER_COND=$(( ${#ARMS[@]} * NUM_SEEDS ))
PER_DATASET=$(( ${#CONDS[@]} * PER_COND ))

DATASET=${DATASETS[$(( IDX / PER_DATASET ))]}
COND=${CONDS[$(( (IDX % PER_DATASET) / PER_COND ))]}
ARM=${ARMS[$(( (IDX % PER_COND) / NUM_SEEDS ))]}
SEED=$(( IDX % NUM_SEEDS ))

echo "========================================"
echo "Array task $IDX: dataset=$DATASET cond=$COND arm=$ARM seed=$SEED"
echo "========================================"

# --cond_params is deliberately left off: the existing baselines were run with
# the defaults, and passing anything here would change the experiment name and
# break the pairing.
case "$ARM" in
  infonoise)
    python -u train.py \
        --dataset "$DATASET" \
        --conditioning "$COND" \
        --train_dist infonoise \
        --dist_params "$GATE_C" \
        --seed "$SEED"
    ;;
  baseline)
    python -u train.py \
        --dataset "$DATASET" \
        --conditioning "$COND" \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  *)
    echo "ERROR: unknown arm '$ARM' at task $IDX" >&2
    exit 1
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX ($DATASET/$COND/$ARM seed $SEED) complete."
echo "========================================"
