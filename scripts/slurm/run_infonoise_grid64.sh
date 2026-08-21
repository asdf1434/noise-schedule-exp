#!/bin/bash

# ==========================================
# InfoNoise generalization grid, eurosat64 only (native 64x64).
#
# Split out from run_infonoise_grid.sh purely because of walltime: 64x64
# training needs 12h where the 28x28/32x32 datasets need 3h, and asking the
# scheduler for 12h on all 80 tasks would cost queue priority for no reason.
# Same grid shape, same arms, same fixed gate pivot -- submit both together.
#
#   1 dataset x 3 conditioning variants x 5 seeds = 15 tasks
#
# eurosat64 is the most interesting cell in the whole grid. It is the ONLY
# dataset where this project measured the sampling-schedule benefit going the
# WRONG way, so if information-guided allocation is really tracking something
# real about the denoising problem rather than reproducing a convention that
# happens to suit MNIST, this is where it should show.
#
# Baselines already exist at 20 seeds for every cell here -- eurosat64 x
# none/lowres/inpaint (run_eurosat64.sh, run_eurosat64_lowres.sh,
# run_eurosat64_inpaint.sh), all with default --cond_params.
#
# Keep GATE_C identical to run_infonoise_grid.sh. Retuning it per dataset
# would confound "InfoNoise generalizes" with "c was retuned per dataset".
#
# Prerequisite: ls -d data/real*/ | grep -i eurosat64
#
# Submit:  sbatch scripts/slurm/run_infonoise_grid64.sh
# ==========================================
#SBATCH --job-name=infonoise_grid64
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/slurm/slurm_infonoise_grid64_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Grid. Task ID decomposes as:
#   dataset_idx = ID / (CONDS * SEEDS),  cond_idx = (ID / SEEDS) % CONDS,
#   seed        = ID % SEEDS
# ==========================================
DATASETS=(eurosat64)
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
