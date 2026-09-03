#!/bin/bash

# ==========================================
# The shifted-regime test: does InfoNoise recover an allocation that a fixed
# schedule gets wrong?
#
# WHY THIS EXPERIMENT. The 2026-08-31 grid found no FID difference between
# InfoNoise and logit_normal(0,1) on mnist / fashion_mnist / cifar10, and the
# profile logs said why: the learned pi ended up within TV 0.04-0.20 of the
# warm-up prior, which IS the baseline. The two arms were training on nearly the
# same distribution, so of course they scored the same. That matches both
# papers -- Raya et al. (arXiv:2602.18647) report InfoNoise "matches established
# baselines on standard image datasets", and their Figure 1b explains it: in
# mature image regimes the inherited allocation already overlaps the informative
# region. The gains they report (2-3x) are all in *shifted* regimes.
#
# So construct a shift with a known answer. Multiplying the data by k = 10 gives
# mmse'(sigma) = k^2 mmse(sigma/k), so the entropy-rate profile
# rho*(sigma) ∝ mmse(sigma)/sigma^2 is the unscaled profile translated by
# exactly a factor of 10 along sigma. Nothing else about the problem changes:
# same images, same architecture, same objective, same sampler. The informative
# region moves one decade and the conventional schedule does not follow it.
#
# Unlike the gamma variants (mnist_g050/g100/g250), which pin the spread at
# _GAMMA_TARGET_STD and so move only the tone distribution, this moves signal
# magnitude and nothing else.
#
# PREDICTIONS, WRITTEN DOWN BEFORE RUNNING. Checked first against a synthetic
# k-scaled Gaussian channel (unit-variance source, mmse_k(sigma) =
# k^2 mmse_1(sigma/k)), driving the real InfoNoiseSampler for 14 refreshes:
#
#   k=1,  gate_c=0.15  ->  pi percentiles 5/50/95 =  0.26 / 1.02 /  5.3
#   k=10, gate_c=1.5   ->                            1.48 / 5.35 / 25.8
#   k=10, gate_c=0.15  ->                            0.57 / 3.83 / 22.4
#   k=10, auto         ->                            0.49 / 3.83 / 22.4
#
#   1. prior does clearly worse than tuned. It is centred at sigma ~ 1 while the
#      information has moved up by a decade.
#   2. auto and c1.5 recover most of that gap without being told where to look.
#   3. The learned pi moves a LOT further from the warm-up prior than the 0.04-0.20
#      total variation measured on unshifted images. If it stays that small here,
#      the estimator is not tracking the data and that is a bug, not a null.
#   4. But pi does NOT shift by the full factor of 10, and expecting it to would be
#      wrong. What shifts by exactly 10 is the target profile rho*; pi = rho*/w,
#      and the loss weight w is fixed in sigma, so the conversion drags the peak
#      back. In this repo's coordinates pi ∝ m_hat(sigma)*gate(sigma)/(1+sigma)^2,
#      and that (1+sigma)^2 is data-independent. The synthetic run shows the median
#      moving 1.02 -> 5.35, about 5x, not 10x. Judge the arms on FID, and read the
#      profile logs as rho*, not pi.
#   5. RISK, already visible above: the auto gate rule derives c = 0.016 on the
#      shifted channel and warns that the pivot sits at the low-noise end of the
#      grid. auto therefore behaves almost identically to an ungated run here
#      (both give 3.83). If auto underperforms c1.5, that is a limitation of the
#      paper's Appendix B.6 onset rule under shift, not of information-guided
#      allocation itself -- report the two separately.
#
# The `tuned` arm uses the naive rescaling mu = log(1/k) = -2.3026, which centres
# t on sigma = 10. That is what a person would try first, not necessarily the
# optimum: the synthetic run suggests pi's own preferred median is nearer sigma
# ~ 5 (mu ~ -1.67). A proper human-search upper bound would sweep mu, which this
# repo can already do cheaply -- worth adding if `tuned` turns out not to win.
#
# ARMS (4 x 10 seeds = 40 tasks, ~42 min each):
#   prior   logit_normal(mu=0, sigma=1)        the misplaced conventional choice,
#                                              and InfoNoise's own warm-up prior
#   tuned   logit_normal(mu=-2.3026, sigma=1)  centred at sigma = 10 by hand:
#                                              sigmoid(-2.3026) = 1/11, so
#                                              sigma = (1-t)/t = 10. This is what
#                                              a human finds by searching, and the
#                                              upper bound InfoNoise should reach
#   auto    infonoise, gate pivot re-derived   the actual claim under test
#   c1.5    infonoise, gate_c = 1.5            pivot rescaled by the same factor
#
# sigma_max is raised to 400 for BOTH InfoNoise arms (default 80). The shifted
# profile's 95th percentile lands near sigma = 68, which the default grid barely
# contains; the estimator would be clipping exactly the region under test. Held
# identical across both arms so it cannot explain a difference between them.
#
# FID: mnist_x10 deliberately reuses mnist's real images and cached stats
# (real_dir=data/real, real_stats_name=mnist_real). Samples are divided by
# DatasetSpec.sample_scale before being written as PNGs, so they land back on
# mnist's scale -- otherwise save_images would clip every pixel to pure black or
# white. No new real-image prep is needed.
#
# Submit (training only -- score afterwards with the usual prep/eval/merge chain,
# passing --dataset mnist_x10):
#   sbatch scripts/slurm/run_mnist_x10.sh
# ==========================================
#SBATCH --job-name=mnist_x10
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
#SBATCH --output=logs/slurm/slurm_mnist_x10_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

ARMS=(prior tuned auto c1.5)
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
echo "Array task $IDX: dataset=mnist_x10 arm=$ARM seed=$SEED"
echo "========================================"

case "$ARM" in
  prior)
    python -u train.py \
        --dataset mnist_x10 \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  tuned)
    python -u train.py \
        --dataset mnist_x10 \
        --train_dist logit_normal \
        --dist_params '{"mu": -2.3026, "sigma": 1.0}' \
        --seed "$SEED"
    ;;
  auto)
    python -u train.py \
        --dataset mnist_x10 \
        --train_dist infonoise \
        --dist_params '{"sigma_max": 400.0}' \
        --seed "$SEED"
    ;;
  c1.5)
    python -u train.py \
        --dataset mnist_x10 \
        --train_dist infonoise \
        --dist_params '{"sigma_max": 400.0, "gate_c": 1.5}' \
        --seed "$SEED"
    ;;
  *)
    echo "ERROR: unknown arm '$ARM' at task $IDX" >&2
    exit 1
    ;;
esac

echo -e "\n========================================"
echo "Array task $IDX (mnist_x10/$ARM seed $SEED) complete."
echo "========================================"
