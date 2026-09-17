#!/bin/bash

# ==========================================
# The k = 1 and k = 0.1 thirds of the CIFAR-10 training-centre axis.
#
# Completes, together with run_cifar_centre_sweep.sh (k = 10, 60 tasks), the
# same three-scale sweep the MNIST side has: run_centre_sweep.sh (k = 1, k = 10)
# plus run_centre_sweep_x01.sh (k = 0.1).
#
# WHY CIFAR AT ALL. MNIST's pixel statistics are nearly binary, which makes its
# m_hat profile unusually sharp. If the shape of the centre curve -- or its
# translation with k -- only holds on MNIST, that is a property of that dataset
# and not of noise allocation. CIFAR-10 is the control: its global sd is 0.4814
# against MNIST's 0.6164, within 28%, so the same sigma grid applies and the two
# datasets' curves are directly comparable against sigma/k.
#
# It is also where the InfoNoise result fails to replicate -- InfoNoise beats
# logit_normal(0,1) on all 9 sampling schedules on mnist_x10 and loses or ties
# on every schedule on cifar10_x10 -- so knowing whether CIFAR's FID responds to
# placement at all decides how to read that non-replication:
#
#   curve peaked and deep -> CIFAR does respond; the non-replication is about
#                            WHERE InfoNoise puts mass. Look at the estimator.
#   curve flat            -> placement is a null on CIFAR with this UNet. The
#                            non-replication is explained, and the follow-up is
#                            model capacity, not the schedule.
#
# t_clip. Same reasoning as run_centre_sweep_x01.sh: the loss weight's dead zone
# sits at sigma <= t_clip/(1 - t_clip), so the default 0.05 puts it at 0.0526,
# which is 0.05k at k = 1 (harmless, well below the data) but 0.53k at k = 0.1
# (inside the informative range). The k = 0.1 arms therefore pass --t_clip 0.005
# for a dead zone at 0.05k, matching k = 1's relative position. REQUIRES the
# --t_clip flag, which is not in commit aa5b0a6 -- pull before submitting.
#
# ARMS (13 x 10 seeds = 130 tasks, ~42 min each). k = 1 is first so a truncated
# array still yields a complete reference curve.
#
#   0-59     cifar10       k=1    sigma = 0.05, 0.1, 0.2, 0.5, 2, 5
#   60-129   cifar10_x0.1  k=0.1  sigma = 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
#
# cifar10 sigma = 1 (mu = 0.0) is NOT repeated -- it already exists at 20 seeds
# and the plot picks it up from master_fid_results.json, which is why k = 1 has
# six arms here and k = 0.1 has seven.
#
# The existing cifar10_x0.1 logit_normal(0,1) runs are at the default t_clip and
# are not reusable; their sigma = 1 is sigma/k = 10, off this grid either way.
# Names here carry _tclip_0.005, so nothing can collide.
#
#   TRAIN=$(sbatch --parsable scripts/slurm/run_cifar_centre_sweep_scales.sh)
#   PREP=$(sbatch --parsable --dependency=afterany:$TRAIN scripts/slurm/run_cifar_sweep_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterany:$PREP scripts/slurm/run_cifar_sweep_eval.sh)
#   sbatch --dependency=afterany:$EVAL scripts/slurm/run_exp1_eval_merge.sh
#
# afterany everywhere, never afterok -- the MNIST centre sweep lost its merge to
# DependencyNeverSatisfied and left 115 trained cells unscored.
#
# run_cifar_sweep_eval_prep.sh / run_cifar_sweep_eval.sh now cover cifar10_x10,
# cifar10 and cifar10_x0.1 (3 datasets x 16 shards, --array=0-47).
# ==========================================
#SBATCH --job-name=cifar_centre_scales
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-129
#SBATCH --output=logs/slurm/slurm_cifar_centre_scales_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

NUM_SEEDS=10

# "<dataset>|<mu>|<t_clip>" -- centre sigma = exp(-mu), width sigma_param = 1.0
ARMS=(
    "cifar10|2.9957|0.05"        # sigma = 0.05   (sigma/k = 0.05)
    "cifar10|2.3026|0.05"        # sigma = 0.1    (sigma/k = 0.1)
    "cifar10|1.6094|0.05"        # sigma = 0.2    (sigma/k = 0.2)
    "cifar10|0.6931|0.05"        # sigma = 0.5    (sigma/k = 0.5)
    "cifar10|-0.6931|0.05"       # sigma = 2      (sigma/k = 2)
    "cifar10|-1.6094|0.05"       # sigma = 5      (sigma/k = 5)
    "cifar10_x0.1|5.2983|0.005"  # sigma = 0.005  (sigma/k = 0.05)
    "cifar10_x0.1|4.6052|0.005"  # sigma = 0.01   (sigma/k = 0.1)
    "cifar10_x0.1|3.9120|0.005"  # sigma = 0.02   (sigma/k = 0.2)
    "cifar10_x0.1|2.9957|0.005"  # sigma = 0.05   (sigma/k = 0.5)
    "cifar10_x0.1|2.3026|0.005"  # sigma = 0.1    (sigma/k = 1)
    "cifar10_x0.1|1.6094|0.005"  # sigma = 0.2    (sigma/k = 2)
    "cifar10_x0.1|0.6931|0.005"  # sigma = 0.5    (sigma/k = 5)
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
REST=${ARM#*|}
MU=${REST%%|*}
T_CLIP=${REST#*|}

echo "========================================"
echo "Array task $IDX: dataset=$DATASET mu=$MU seed=$SEED t_clip=$T_CLIP"
echo "  centre sigma = exp(-mu)"
echo "========================================"

python -u train.py \
    --dataset "$DATASET" \
    --train_dist logit_normal \
    --dist_params "{\"mu\": $MU, \"sigma\": 1.0}" \
    --t_clip "$T_CLIP" \
    --seed "$SEED"

echo -e "\n========================================"
echo "Array task $IDX ($DATASET/mu=$MU/seed$SEED) complete."
echo "========================================"
