#!/bin/bash

# ==========================================
# The shifted-regime test, downward: mnist_x10's mirror image.
#
# WHY BOTH DIRECTIONS. run_mnist_x10.sh multiplies the data by k = 10, which
# translates the entropy-rate profile rho*(sigma) ∝ mmse(sigma)/sigma^2 up by
# exactly one decade (mmse'(sigma) = k^2 mmse(sigma/k)). This script does the
# same with k = 0.1, moving it down by one decade instead. Running only the
# upward shift leaves an obvious alternative explanation open: that InfoNoise
# helps whenever sigma is large, or that raising sigma_max is what did the work.
# A symmetric result in the opposite direction rules that out. A result that
# appears in one direction only is itself informative, and worth knowing before
# any of this is written up.
#
# PREDICTIONS, WRITTEN DOWN BEFORE RUNNING. By the same algebra as x10, with the
# signs flipped: the informative region moves to sigma ~ 0.1, the conventional
# prior centred at sigma ~ 1 is again misplaced, and pi should move a long way
# from the warm-up prior (far more than the TV 0.04-0.20 seen on unshifted
# images). As in x10, pi will NOT move by the full factor of 10: pi = rho*/w and
# the loss weight w is fixed in sigma, so the (1+sigma)^2 conversion drags the
# peak back. Judge the arms on FID and read the profile logs as rho*, not pi.
#
# NOT YET CHECKED ON THE SYNTHETIC CHANNEL. The x10 predictions were validated
# first against a k-scaled Gaussian channel driving the real InfoNoiseSampler.
# The equivalent sweep for k = 0.1 has not been run, so the numbers above are
# the algebra's prediction rather than a measured one. Worth doing before
# trusting a null result from this script.
#
# ARMS (4 x 10 seeds = 40 tasks):
#   prior   logit_normal(mu=0, sigma=1)        the misplaced conventional choice,
#                                              and InfoNoise's own warm-up prior
#   tuned   logit_normal(mu=+2.3026, sigma=1)  centred at sigma = 0.1 by hand:
#                                              sigmoid(2.3026) = 10/11, so
#                                              sigma = (1-t)/t = 0.1. The mirror
#                                              of x10's mu = -2.3026
#   auto    infonoise, gate pivot re-derived   the actual claim under test
#   c0.015  infonoise, gate_c = 0.015          the default 0.15 scaled by k, as
#                                              x10 scaled it to 1.5
#
# GRID BOUNDARY. x10 raised sigma_max 80 -> 400 because the shifted profile ran
# off the top of the estimator's grid. The downward shift has the mirror problem
# at the bottom, so sigma_min drops 0.002 -> 0.0002 and sigma_max stays at its
# default. This keeps roughly the same headroom below the profile's 5th
# percentile that the unshifted runs had. Held identical across both InfoNoise
# arms so it cannot explain a difference between them.
#
# WATCH THE GATE. The auto gate-pivot rule needs a resolved low-noise boundary
# segment to fire correctly, and it already warned on x10's upward shift. Moving
# the whole profile toward sigma_min puts more pressure on exactly that rule, so
# a warning here is likely. If auto underperforms c0.015, check the profile logs
# for the pivot warning before concluding anything about information-guided
# allocation -- that would be a limitation of the paper's Appendix B.6 onset
# rule, not of the method.
#
# FID: like mnist_x10, this reuses mnist's real images and cached stats
# (real_dir=data/real, real_stats_name=mnist_real). Samples are multiplied back
# up by 1/sample_scale before being written as PNGs. No new real-image prep.
#
# Submit (training only -- score afterwards with the usual prep/eval/merge chain,
# passing --dataset mnist_x0.1):
#   sbatch scripts/slurm/run_mnist_x0.1.sh
# ==========================================
#SBATCH --job-name=mnist_x0.1
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/slurm/slurm_mnist_x0.1_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

ARMS=(prior tuned auto c0.015)
NUM_SEEDS=10

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

echo "========================================"
echo "Array task $IDX: dataset=mnist_x0.1 arm=$ARM seed=$SEED"
echo "========================================"

case "$ARM" in
  prior)
    python -u train.py \
        --dataset mnist_x0.1 \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  tuned)
    python -u train.py \
        --dataset mnist_x0.1 \
        --train_dist logit_normal \
        --dist_params '{"mu": 2.3026, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  auto)
    python -u train.py \
        --dataset mnist_x0.1 \
        --train_dist infonoise \
        --dist_params '{"sigma_min": 0.0002}' \
        --seed "$SEED"
    ;;
  c0.015)
    python -u train.py \
        --dataset mnist_x0.1 \
        --train_dist infonoise \
        --dist_params '{"sigma_min": 0.0002, "gate_c": 0.015}' \
        --seed "$SEED"
    ;;
  *)
    echo "ERROR: unknown arm '$ARM' at task $IDX" >&2
    exit 1
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX (mnist_x0.1/$ARM seed $SEED) complete."
echo "========================================"
