#!/bin/bash

# ==========================================
# Where is the best place to put training noise, and does it move with the data?
#
# WHY. Every claim so far has assumed the FID-optimal training distribution sits
# where the entropy-rate profile says it should. That assumption has never been
# measured. What IS proven is only the translation identity: for x' = kx,
# mmse_k(sigma) = k^2 mmse_1(sigma/k), so per unit log-sigma the entropy-rate
# target rho*_k(sigma) = rho*_1(sigma/k) -- a pure translation by log k, same
# shape, same height. Whether training THERE minimizes FID is a modelling
# assumption from the paper, not a theorem.
#
# On mnist_x10 we have sampled exactly three placements (sigma = 1, ~2, 10) and
# they came out within 1.3x of each other, so we cannot say where the optimum
# is, how sharp it is, or whether InfoNoise's choice is near it.
#
# THIS RUN sweeps the centre of a fixed-width logit-normal across two decades at
# two data scales, at matched sigma/k. Everything else is held constant:
# logit_normal(mu = -ln sigma, sigma_param = 1.0).
#
#   sigma/k      0.05    0.1    0.2    0.5      1      2      5
#   k=1   sigma  0.05    0.1    0.2    0.5      1      2      5
#   k=10  sigma  0.5     1      2      5        10     20     50
#
# THE TEST. If the identity holds AND rho* is FID-optimal, the k=10 curve must
# be the k=1 curve translated by exactly one decade -- i.e. the two curves lie
# on top of each other when plotted against sigma/k.
#
#   curves coincide  -> theory validated; InfoNoise's failure to track the shift
#                       (rho_hat moved 5.3x of the required 10x, all of the
#                       shortfall attributable to the auto gate cutoff) is a
#                       real defect worth reporting as one.
#   curves differ    -> the entropy-rate target is not what FID wants. InfoNoise
#                       is optimizing the wrong objective, and its sitting near
#                       sigma ~ 2 regardless of k is correct by accident. Would
#                       also explain why the MNIST win does not replicate on
#                       CIFAR-10 (prior better on 5 of 9 schedules there).
#   both curves flat -> placement does not matter on this task at all. The whole
#                       axis is a null and can be reported as one, with evidence
#                       rather than inference.
#
# All three outcomes are publishable, which is the point: unlike a two-arm
# comparison this cannot come back ambiguous.
#
# ARMS (12 x 10 seeds = 120 tasks, ~42 min each). mnist_x10 already has
# sigma = 1 (mu 0.0) and sigma = 10 (mu -2.3026) at 10 seeds, so only five of
# its seven centres are new. Array order puts mnist_x10 first.
#
#   0-49    mnist_x10  sigma = 0.5, 2, 5, 20, 50
#   50-119  mnist      sigma = 0.05, 0.1, 0.2, 0.5, 1, 2, 5
#
# mnist's existing logit_normal runs use the legacy naming scheme and a
# different sweep, so all seven centres are rerun here to keep the two curves
# internally consistent.
#
#   TRAIN=$(sbatch --parsable scripts/slurm/run_centre_sweep.sh)
#   PREP=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm/run_sweep_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_sweep_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=centre_sweep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/slurm/slurm_centre_sweep_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

NUM_SEEDS=10

# "<dataset>|<mu>"  -- centre sigma = exp(-mu), width fixed at sigma_param = 1.0
ARMS=(
    "mnist_x10|0.6931"    # sigma = 0.5    (sigma/k = 0.05)
    "mnist_x10|-0.6931"   # sigma = 2      (sigma/k = 0.2)
    "mnist_x10|-1.6094"   # sigma = 5      (sigma/k = 0.5)
    "mnist_x10|-2.9957"   # sigma = 20     (sigma/k = 2)
    "mnist_x10|-3.9120"   # sigma = 50     (sigma/k = 5)
    "mnist|2.9957"        # sigma = 0.05
    "mnist|2.3026"        # sigma = 0.1
    "mnist|1.6094"        # sigma = 0.2
    "mnist|0.6931"        # sigma = 0.5
    "mnist|0.0"           # sigma = 1
    "mnist|-0.6931"       # sigma = 2
    "mnist|-1.6094"       # sigma = 5
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

DATASET=${ARM%%|*}
MU=${ARM#*|}

echo "========================================"
echo "Array task $IDX: dataset=$DATASET mu=$MU seed=$SEED"
echo "  centre sigma = exp(-mu)"
echo "========================================"

python -u train.py \
    --dataset "$DATASET" \
    --train_dist logit_normal \
    --dist_params "{\"mu\": $MU, \"sigma\": 1.0}" \
    --seed "$SEED"

echo -e "\n========================================"
echo "Array task $IDX ($DATASET/mu=$MU/seed$SEED) complete."
echo "========================================"
