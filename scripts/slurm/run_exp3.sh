#!/bin/bash

# ==========================================
# Experiment 3: finer sigma sweep for logit_normal(mu=0, sigma) training
# distributions, to find where between the two known anchors -- sigma=1.0
# (exp1: no significant effect vs. uniform) and sigma=0.3 (exp2: dramatically
# worse than uniform on every sampling schedule, p<1e-5) -- peakedness starts
# to hurt FID, and whether the transition is sharp or gradual.
#
# 8 sigma values x 20 seeds = 160 independent tasks, run in PARALLEL across
# GPUs (same job-array pattern as run_exp2.sh). Both anchors (0.3 and 1.0)
# are re-run here with the same 20-seed budget as the six new interior
# points, so every sigma in the sweep has a balanced, directly comparable
# sample size (exp2's sigma=0.3 runs and exp1's sigma=1.0 runs are NOT
# reused). If reusing exp2's existing sigma=0.3 n=20 runs is acceptable
# instead, submit with `--array=20-159` to skip the redundant first 20
# tasks.
# ==========================================
#SBATCH --job-name=exp3_sigma_sweep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
# Dropped vision-shared-rtx2080ti, vision-shared-titanrtx, and
# vision-shared-v100 (Turing/Volta) after confirming on beery-a100-1 (0MiB
# used, no other processes) that a completely idle old-arch-class GPU still
# hits the cuDNN status-5003 conv failure while an idle A100 runs clean --
# this is an architecture incompatibility, not contention, so excluding
# individual nodes was never going to fully fix it. rtx3090/rtx3080 are
# also Ampere like a100 and kept since they should behave the same way.
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
# Union of every bad node found across all slurm scripts to date:
# isola-2080ti-4, gpu19-2.drl, gpu20-2.drl (reliable cuInit failures, found in
# run_exp2.sh) plus improbablex002, isola-ada6000-1, and gpu19-1.drl (found
# later during the EuroSAT runs -- gpu19-1.drl was thought to be fine when
# run_exp2.sh was written, but run_eurosat.sh later excluded it too).
#
# gpu20-3.drl added here after job 1123254: 80 cuDNN-autotune failures,
# far more than any other node on a partition (vision-shared-rtx2080ti)
# that's otherwise still in use -- looks like a genuinely broken node on
# top of the architecture issue below, so excluded explicitly. The other
# repeat offenders from that job (agrawal-v100-1, freeman-v100-1,
# torralba-v100-1/2, isola-v100-2, freeman-titanrtx-2, up to 80
# failures each) don't need individual --exclude entries: they're all
# v100/titanrtx nodes, already unreachable now that vision-shared-v100 and
# vision-shared-titanrtx were dropped from --partition above.
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2
# Auto-resubmit a task if SLURM kills it (preemption, node failure) instead
# of requiring a manual requeue_failed.sh pass. Doesn't distinguish
# contention/preemption from a genuine code bug -- if train.py itself is
# broken, this will retry it fruitlessly, so still check logs occasionally.
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-159
#SBATCH --output=logs/slurm/slurm_exp3_sigma_sweep_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# Every observed failure in this job is the identical cuDNN autotune error
# ("Failed to determine best cudnn convolution algorithm", status 5003) for
# the same conv shape -- and critically, it now reproduces on EVERY node
# (not just old-arch/flaky ones), including at actual kernel execution
# time, not just during algorithm autotuning. That rules out "one bad
# node"; a uniform failure across all node types on a heavily shared
# cluster points at GPU memory pressure instead: JAX preallocates ~90% of
# a GPU's VRAM by default the instant it touches the device, which can
# starve cuDNN's conv kernels if SLURM's GPU isolation lets another
# process share the same physical GPU. Disabling preallocation makes JAX
# grow its allocation on demand instead, which is the standard fix for
# this failure mode on shared clusters.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

# ==========================================
# Map array task ID -> (sigma, seed)
# 8 sigmas x 20 seeds each: sigma_idx = ID / 20, seed = ID % 20
# ==========================================
NUM_SEEDS=20

SIGMAS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

IDX=$SLURM_ARRAY_TASK_ID
SIGMA_IDX=$((IDX / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))
SIGMA=${SIGMAS[$SIGMA_IDX]}

echo "========================================"
echo "Array task $IDX: sigma=$SIGMA seed=$SEED"
echo "========================================"

python -u train.py \
    --train_dist logit_normal \
    --dist_params "{\"mu\": 0.0, \"sigma\": $SIGMA}" \
    --seed "$SEED"

echo -e "\n========================================"
echo "Array task $IDX (sigma=$SIGMA seed=$SEED) complete."
echo "Once ALL 160 array tasks finish, run evaluate_fid.py separately -- reuse"
echo "run_exp1_eval_array.sh + run_exp1_eval_merge.sh (they walk all of"
echo "eval_runs/ generically, no changes needed for this experiment), then"
echo "python scripts/plots/aggregate_fid.py and"
echo "python scripts/plots/plot_sigma_sweep.py."
echo "========================================"
