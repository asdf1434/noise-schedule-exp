#!/bin/bash

# ==========================================
# Does the training-noise centre matter on CIFAR-10 at 10x, the way it does on
# MNIST at 10x?
#
# WHY. InfoNoise beats the default logit-normal(0,1) on mnist_x10 across all 9
# inference schedules (14-46%), and loses or ties on every schedule on
# cifar10_x10. That is not a placement difference: the two datasets' signal
# scales are within 28% of each other (global sd 0.6164 vs 0.4814), so at k=10
# both want their training mass near sigma ~ 5, the default sits at sigma = 1
# for both, and InfoNoise moves partway up on both (peak sigma 2.29 vs 1.71).
#
# The leading explanation is that CIFAR's FID barely responds to noise
# allocation at all. Evidence so far is indirect: sweeping the INFERENCE
# schedule (a different knob, same kind) spans 53.8 FID points on mnist_x10 and
# only 14.7 on cifar10_x10. This run measures the training-centre axis directly.
#
# THE TEST. Same centres as the mnist_x10 arm of run_centre_sweep.sh, so the two
# curves are directly comparable against sigma/k.
#
#   sigma/k       0.05   0.1    0.2    0.5     1      2      5
#   k=10  sigma   0.5    1      2      5       10     20     50
#
#   curve peaked and deep -> CIFAR does respond; InfoNoise's non-replication is
#                            about WHERE it puts the mass, not whether placement
#                            matters. Look at the estimator next.
#   curve flat            -> placement is a null on CIFAR with this UNet. The
#                            non-replication is explained, and the follow-up is
#                            model capacity (scale the UNet until CIFAR's FID
#                            reaches MNIST's range, then rerun).
#
# ARMS (6 x 10 seeds = 60 tasks). cifar10_x10 already has sigma = 1 (mu 0.0) at
# 10 seeds from the gridscale run, so that centre is NOT repeated here -- the
# plot picks it up from master_fid_results.json.
#
#   0-9     sigma = 0.5    (mu  0.6931)
#   10-19   sigma = 2      (mu -0.6931)
#   20-29   sigma = 5      (mu -1.6094)
#   30-39   sigma = 10     (mu -2.3026)
#   40-49   sigma = 20     (mu -2.9957)
#   50-59   sigma = 50     (mu -3.9120)
#
# NOTE on the chain: use afterany, not afterok, between train and prep. Both
# array jobs run so far lost tasks to node-level flakes (5 of 120 in the centre
# sweep, 10 of 32 in its eval array), and afterok turned each into a merge that
# never ran and zero scored results. Every later stage skips completed work, so
# a partial train array should degrade to partial results, not none.
#
#   TRAIN=$(sbatch --parsable scripts/slurm/run_cifar_centre_sweep.sh)
#   PREP=$(sbatch --parsable --dependency=afterany:$TRAIN scripts/slurm/run_cifar_sweep_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_cifar_sweep_eval.sh)
#   sbatch --dependency=afterany:$EVAL scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=cifar_centre_sweep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1,torralba-3090-3,improbablex001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/slurm/slurm_cifar_centre_sweep_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

NUM_SEEDS=10
DATASET=cifar10_x10

# centre sigma = exp(-mu), width fixed at sigma_param = 1.0
MUS=(
    "0.6931"    # sigma = 0.5    (sigma/k = 0.05)
    "-0.6931"   # sigma = 2      (sigma/k = 0.2)
    "-1.6094"   # sigma = 5      (sigma/k = 0.5)
    "-2.3026"   # sigma = 10     (sigma/k = 1)
    "-2.9957"   # sigma = 20     (sigma/k = 2)
    "-3.9120"   # sigma = 50     (sigma/k = 5)
)

TOTAL=$(( ${#MUS[@]} * NUM_SEEDS ))
EXPECTED_MAX=$(( TOTAL - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds the grid" \
         "(${#MUS[@]} centres x $NUM_SEEDS seeds = $TOTAL)." \
         "Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
MU=${MUS[$(( IDX / NUM_SEEDS ))]}
SEED=$(( IDX % NUM_SEEDS ))

echo "========================================"
echo "Array task $IDX: dataset=$DATASET mu=$MU seed=$SEED"
echo "  centre sigma = exp(-mu)"
echo "========================================"

python -u train.py \
    --dataset "$DATASET" \
    --train_dist logit_normal \
    --dist_params "{\"mu\": $MU, \"sigma\": 1.0}" \
    --seed "$SEED"
