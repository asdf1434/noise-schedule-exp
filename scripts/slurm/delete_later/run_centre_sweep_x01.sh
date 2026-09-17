#!/bin/bash

# ==========================================
# The missing bottom third of the training-centre axis: k = 0.1.
#
# WHY. run_centre_sweep.sh swept the centre of a fixed-width logit-normal at
# k = 1 (mnist) and k = 10 (mnist_x10) at matched sigma/k. The test is whether
# the two curves coincide when plotted against sigma/k -- the translation
# identity mmse_k(sigma) = k^2 mmse_1(sigma/k) says the entropy-rate target
# translates by exactly log k, so if that target is what FID wants, the curves
# must lie on top of each other. Two points cannot distinguish "translates
# correctly" from "translates by the wrong amount in a way two scales happen to
# agree on". This adds the third, and it is the one that shifts DOWN.
#
# WHY IT COULD NOT BE RUN BEFORE. The objective's loss weight
# w = 1/max(t_clip, 1-t)^2 stops growing once 1 - t <= t_clip, which in sigma
# coordinates is a dead zone at sigma <= t_clip/(1 - t_clip) = 0.0526 at the
# default t_clip = 0.05. That default assumes unit-amplitude data. At k = 0.1
# the informative sigma range translates down one decade and lands inside it:
# measured on the existing mnist_x0.1 InfoNoise runs, 51% of the training mass
# sits at sigma <= 0.0526, against 8.9% on mnist_x10. Half of every k = 0.1 run
# so far trained against a clipped objective, which is the most likely cause of
# the FID ~200 plateau on 7 of 9 sampling schedules noted in
# docs/updates/UPDATE_david_2026-09-03.md.
#
# So every arm here passes --t_clip 0.005, giving a dead zone at sigma <= 0.00503
# = 0.05k -- the same position relative to the data that k = 1 has at the
# default (0.0526 = 0.05k). The k = 10 arm keeps t_clip = 0.05, whose dead zone
# sits at 0.005k, a decade FURTHER below its informative range. The three arms
# are therefore not at an identical t_clip, but all three have the dead zone
# safely below where the data lives, which is what the comparison needs. Matching
# k = 10 exactly would mean t_clip = 0.345, clipping everything below sigma = 1
# -- worse, not more comparable.
#
# REQUIRES the --t_clip flag (src/loss.py, src/utils.py, train.py). It is not in
# commit aa5b0a6; pull before submitting or every task dies on an unknown arg.
#
# NOTE the existing mnist_x0.1 logit_normal runs (mu = 0.0 and mu = 2.3026,
# 10 seeds each) are at the default t_clip and are NOT reusable here. The name
# carries _tclip_0.005, so these write to fresh paths and cannot collide.
#
# ARMS (7 centres x 10 seeds = 70 tasks, ~42 min each), matched to the sigma/k
# grid the other two scales already use:
#
#   sigma/k     0.05    0.1    0.2    0.5      1      2      5
#   k=0.1 sigma 0.005   0.01   0.02   0.05     0.1    0.2    0.5
#   mu=-ln sig  5.2983  4.6052 3.9120 2.9957   2.3026 1.6094 0.6931
#
#   TRAIN=$(sbatch --parsable scripts/slurm/run_centre_sweep_x01.sh)
#   PREP=$(sbatch --parsable --dependency=afterany:$TRAIN scripts/slurm/run_sweep_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterany:$PREP scripts/slurm/run_sweep_eval.sh)
#   sbatch --dependency=afterany:$EVAL scripts/slurm/run_exp1_eval_merge.sh
#
# afterany everywhere, never afterok: the MNIST centre sweep lost its merge to
# DependencyNeverSatisfied when 10 of 32 eval shards hit a bad node, stranding
# 115 trained cells unscored.
#
# run_sweep_eval_prep.sh / run_sweep_eval.sh now cover mnist, mnist_x10 and
# mnist_x0.1 (3 datasets x 16 shards, --array=0-47).
# ==========================================
#SBATCH --job-name=centre_sweep_x01
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-69
#SBATCH --output=logs/slurm/slurm_centre_sweep_x01_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

NUM_SEEDS=10

# Dead zone sigma <= t_clip/(1 - t_clip) = 0.00503 = 0.05k, matching the
# position k = 1 has at the default. See the header.
T_CLIP=0.005

# centre sigma = exp(-mu), width fixed at sigma_param = 1.0
MUS=(
    "5.2983"   # sigma = 0.005  (sigma/k = 0.05)
    "4.6052"   # sigma = 0.01   (sigma/k = 0.1)
    "3.9120"   # sigma = 0.02   (sigma/k = 0.2)
    "2.9957"   # sigma = 0.05   (sigma/k = 0.5)
    "2.3026"   # sigma = 0.1    (sigma/k = 1)
    "1.6094"   # sigma = 0.2    (sigma/k = 2)
    "0.6931"   # sigma = 0.5    (sigma/k = 5)
)

TOTAL=$(( ${#MUS[@]} * NUM_SEEDS ))
EXPECTED_MAX=$(( TOTAL - 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_MAX" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID exceeds the grid" \
         "(${#MUS[@]} arms x $NUM_SEEDS seeds = $TOTAL)." \
         "Set --array=0-$EXPECTED_MAX in this file." >&2
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
MU=${MUS[$(( IDX / NUM_SEEDS ))]}
SEED=$(( IDX % NUM_SEEDS ))

echo "========================================"
echo "Array task $IDX: dataset=mnist_x0.1 mu=$MU seed=$SEED t_clip=$T_CLIP"
echo "  centre sigma = exp(-mu)"
echo "========================================"

python -u train.py \
    --dataset mnist_x0.1 \
    --train_dist logit_normal \
    --dist_params "{\"mu\": $MU, \"sigma\": 1.0}" \
    --t_clip "$T_CLIP" \
    --seed "$SEED"

echo -e "\n========================================"
echo "Array task $IDX (mnist_x0.1/mu=$MU/seed$SEED) complete."
echo "========================================"
