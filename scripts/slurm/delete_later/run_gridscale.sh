#!/bin/bash

# ==========================================
# Run A of the InfoNoise scale-shift follow-up: give the estimator a noise grid
# that scales with the data, on two datasets and both directions of shift.
#
# Supersedes run_x10_gridscale.sh, which covered mnist_x10 alone.
#
# WHY. InfoNoise's automatic gate rule (Appendix B.6) scans
# r(sigma) = m_hat/sigma^3, normalizes by max r, and takes the largest sigma
# with r/max(r) >= p (p=0.002). Because 1/sigma^3 diverges, max r is attained at
# the grid's LOW EDGE rather than at any feature of the data -- verified in the
# original mnist and mnist_x10 runs (argmax at sigma=0.002, the first bin). So
# the cutoff is anchored to sigma_min, not to the data.
#
# run_mnist_x10.sh raised sigma_max 80 -> 400 but left sigma_min at the default
# 0.002. Relative to the data that grid reaches down to sigma/k = 2e-4, a decade
# deeper than mnist's 2e-3, where r is ~1000x larger. The normalizer inflates,
# the threshold drops, and the cutoff came out at 0.0175 instead of the ~0.21
# that would be mnist's 0.0207 scaled by k.
#
# Replaying the rule offline on the x10 logs, restricted to mnist's grid scaled
# by 10 ([0.0204, 784]), gives 0.427 against a target of 0.207 -- within ~2x
# instead of ~12x off. That is the prediction this job tests live.
#
# sigma_min=0.002 / sigma_max=80 is EDM's standard range and assumes data at
# roughly unit scale. Scaling the data breaks that assumption, so this may be an
# unstated precondition of the rule rather than a defect in it. Which of the two
# it is changes what gets reported.
#
# THE GRID RULE HERE. For a dataset scaled by k, use [0.002k, 80k] -- the
# default grid translated by exactly the amount the entropy-rate profile
# translates (mmse_k(sigma) = k^2 mmse_1(sigma/k)).
#
# ARMS (6 x 10 seeds = 60 tasks, ~42 min each):
#   0-9    mnist_x10     infonoise, grid [0.02, 800]
#   10-19  mnist_x0.1    infonoise, grid [0.0002, 8]
#   20-29  cifar10_x10   infonoise, grid [0.02, 800]
#   30-39  cifar10_x10   logit_normal(0, 1)          <- baseline, does not exist yet
#   40-49  cifar10_x0.1  infonoise, grid [0.0002, 8]
#   50-59  cifar10_x0.1  logit_normal(0, 1)          <- baseline, does not exist yet
#
# The two mnist arms need no baseline task: the logit_normal(0, 1) arm already
# exists at 10 seeds for both mnist_x10 and mnist_x0.1 (from run_mnist_x10.sh).
# The scaled CIFAR datasets are new, so each needs its own baseline.
#
# CIFAR-10 is the second dataset because MNIST's pixel statistics are nearly
# binary, which makes its m_hat profile unusually sharp. If the result only
# holds on MNIST it is a property of that dataset, not of the method.
#
# The gate pivot is left auto-derived everywhere. Nothing else changes: same
# architecture, objective, sampler, seeds.
#
# PREDICTIONS, WRITTEN DOWN BEFORE RUNNING:
#   1. Derived gate cutoff lands near 0.4 on the x10 arms (offline replay said
#      0.427), not 0.0175. Read it off gate_c in the first refresh of any task.
#   2. The gate-pivot WARNING that fired on all ten original auto runs does not
#      fire here.
#   3. The mode of rho_hat moves toward k but does NOT reach it, because the
#      loss weight's fixed scale at sigma=1 still pins the sampled density (see
#      docs/updates/UPDATE_david_2026-09-03.md). Removing that weight is a separate run.
#   4. FID improves on the original auto arm. This is the weakest prediction --
#      the mechanism checks (1)-(3) read off the profile logs alone.
#
# KNOWN RISK. The existing mnist_x0.1 runs sit at FID ~200 on 7 of 9 sampling
# schedules with no arm separating, cause not yet confirmed (likely the x10
# rescale applied before samples are written to PNG). cifar10_x0.1 may inherit
# the same problem. The x10 arms are the ones to trust if so.
#
# Submit training, then score with the usual chain (run_shift_eval*.sh cover all
# four datasets; evaluate_fid.py skips cells already scored, so the rerun only
# picks up the new experiments):
#   TRAIN=$(sbatch --parsable scripts/slurm/run_gridscale.sh)
#   PREP=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm/run_shift_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_shift_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
# or in one go:
#   scripts/slurm/run_pipeline.sh scripts/slurm/run_gridscale.sh scripts/slurm/run_shift_eval_prep.sh
# ==========================================
#SBATCH --job-name=gridscale
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/slurm/slurm_gridscale_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

NUM_SEEDS=10

# One entry per arm, in array order: "<dataset> <train_dist> <dist_params>".
ARMS=(
    "mnist_x10     infonoise    {\"sigma_min\": 0.02, \"sigma_max\": 800.0}"
    "mnist_x0.1    infonoise    {\"sigma_min\": 0.0002, \"sigma_max\": 8.0}"
    "cifar10_x10   infonoise    {\"sigma_min\": 0.02, \"sigma_max\": 800.0}"
    "cifar10_x10   logit_normal {\"mu\": 0.0, \"sigma\": 1.0}"
    "cifar10_x0.1  infonoise    {\"sigma_min\": 0.0002, \"sigma_max\": 8.0}"
    "cifar10_x0.1  logit_normal {\"mu\": 0.0, \"sigma\": 1.0}"
)

TOTAL=$(( ${#ARMS[@]} * NUM_SEEDS ))
EXPECTED_MAX=$(( TOTAL - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds the grid" \
         "(${#ARMS[@]} arms x $NUM_SEEDS seeds = $TOTAL)." \
         "Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( IDX / NUM_SEEDS ))]}
SEED=$(( IDX % NUM_SEEDS ))

DATASET=$(echo "$ARM" | awk '{print $1}')
DIST=$(echo "$ARM" | awk '{print $2}')
DIST_PARAMS=$(echo "$ARM" | cut -d'{' -f2- | sed 's/^/{/')

echo "========================================"
echo "Array task $IDX: dataset=$DATASET dist=$DIST seed=$SEED"
echo "  dist_params=$DIST_PARAMS"
echo "========================================"

python -u train.py \
    --dataset "$DATASET" \
    --train_dist "$DIST" \
    --dist_params "$DIST_PARAMS" \
    --seed "$SEED"

echo -e "\n========================================"
echo "Array task $IDX ($DATASET/$DIST/seed$SEED) complete."
echo "========================================"
