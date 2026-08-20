#!/bin/bash

# ==========================================
# InfoNoise (arXiv:2602.18647): does an online, information-guided training
# noise distribution beat the fixed one it starts from?
#
# 4 gate settings x 10 seeds = 40 independent tasks, run in PARALLEL across
# GPUs (same job-array pattern as run_exp1.sh / run_exp2.sh). Each task does
# ONE training run (~42 min at ~25s/epoch x 100 epochs).
#
# WHY GATE_C IS THE SWEPT AXIS. InfoNoise has exactly one hyperparameter that
# can silently wreck a run: the endpoint gate pivot c. The estimated profile
# carries a 1/sigma^3 path factor, so it blows up at the low-noise end where
# the estimate is least trustworthy, and the gate sigma^3/(sigma^3 + c^3) is
# what suppresses that. Get c wrong and pi collapses onto sigma -> 0. The
# paper's own practice is to pin c per domain (c = 0.2 for its DNA runs,
# c ~ 0.15 for CIFAR-10), and we have never seen a real profile from THIS
# setup, so the sweep covers the gate rather than assuming a value:
#
#   auto    no --dist_params; c re-derived each refresh by the paper's
#           onset-of-information rule (p = 0.002, Appendix B.6)
#   off     gate disabled entirely -- the ablation that says whether the
#           gate does any work here at all
#   c015    pinned at the paper's CIFAR-10 value
#   c040    pinned higher, pushing allocation further from the clean end
#
# Everything else is held fixed: model, objective, loss weighting, optimizer,
# inference sampler, eval budget. Only the training noise distribution varies,
# which is the comparison the paper's experimental design calls for.
#
# BASELINE -- NOT RERUN HERE. InfoNoise's warm-up prior is logit_normal(mu=0,
# sigma=1), so the honest control is that same distribution held fixed for the
# whole run. Those mnist runs already exist from exp 1; reuse them (see
# results/aggregated_fid_results.json) rather than spending GPU-hours on them
# again. Seeds 0-9 here overlap exp 1's seed range so the arms pair up per
# seed. The train.py refactor that added InfoNoise keeps the noise stream
# bit-identical for a given --seed, so exp 1's runs remain directly comparable
# -- a fixed-dist run reproduces its old loss exactly.
#
# If exp 1's logit_normal(0,1) mnist runs are NOT on disk, add "baseline" to
# ARMS below and widen --array to match (see the guard in the body).
#
# Submit the whole overnight chain (train -> stats+shard seeding -> FID ->
# merge) in one command:
#
#   scripts/slurm/run_pipeline.sh scripts/slurm/run_infonoise.sh \
#       scripts/slurm/run_infonoise_eval_prep.sh
#
# Or just this stage:  sbatch scripts/slurm/run_infonoise.sh
# ==========================================
#SBATCH --job-name=infonoise
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# Same known-bad nodes excluded as run_exp2.sh (CUDA init failures).
#SBATCH --exclude=andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/slurm/slurm_infonoise_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# Map array task ID -> (arm, seed)
# arm_idx = ID / NUM_SEEDS, seed = ID % NUM_SEEDS
# ==========================================
NUM_SEEDS=10

ARMS=(auto off c015 c040)

# Fail at submit time rather than silently running a truncated or
# out-of-range sweep -- an index past the end of ARMS would otherwise pick an
# empty arm name and fall through the case statement doing nothing, exiting 0
# and looking like a success in squeue.
EXPECTED_MAX=$(( ${#ARMS[@]} * NUM_SEEDS - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds the grid of ${#ARMS[@]} arms" \
         "x $NUM_SEEDS seeds. Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
ARM_IDX=$((IDX / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
ARM=${ARMS[$ARM_IDX]}

echo "========================================"
echo "Array task $IDX: arm=$ARM seed=$SEED"
echo "========================================"

# Watch the per-refresh lines in this log. Two things matter:
#   - the sigma/t quantiles of pi should move off the warm-up prior over the
#     first few refreshes, then settle (the paper's check is that consecutive
#     sampler-CDF changes fall below 1e-2)
#   - "WARNING: infonoise auto gate pivot" means c collapsed to the low-noise
#     end of the grid and pi is probably degenerate. It can only appear in the
#     "auto" arm (the pinned arms never resolve c), and there it is a finding
#     rather than a bug: it says the onset rule found no boundary structure to
#     gate, which is itself the answer about whether the rule transfers here.
case "$ARM" in
  auto)
    python -u train.py \
        --train_dist infonoise \
        --seed "$SEED"
    ;;
  off)
    python -u train.py \
        --train_dist infonoise \
        --dist_params '{"gate_c": 0.0}' \
        --seed "$SEED"
    ;;
  c015)
    python -u train.py \
        --train_dist infonoise \
        --dist_params '{"gate_c": 0.15}' \
        --seed "$SEED"
    ;;
  c040)
    python -u train.py \
        --train_dist infonoise \
        --dist_params '{"gate_c": 0.4}' \
        --seed "$SEED"
    ;;
  baseline)
    # Only needed if exp 1's runs aren't on disk. Same distribution InfoNoise
    # warms up from, held fixed for the whole run.
    python -u train.py \
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
echo "Array task $IDX ($ARM seed $SEED) complete."
echo ""
echo "Once ALL ${#ARMS[@]}x$NUM_SEEDS tasks finish:"
echo "  1. FID:      scripts/slurm/run_infonoise_eval_prep.sh, then"
echo "               run_exp1_eval_array.sh + run_exp1_eval_merge.sh"
echo "               (run_pipeline.sh chains all three for you)"
echo "  2. Profile:  python -m scripts.plots.plot_infonoise_profile \\"
echo "                   --experiment ds-mnist__cond-none__dist-infonoise__seed-0"
echo "               Check rho_hat is a single interior hump, not a wall"
echo "               against either edge of the sigma grid."
echo "  3. Compare:  python scripts/plots/aggregate_fid.py"
echo "========================================"
