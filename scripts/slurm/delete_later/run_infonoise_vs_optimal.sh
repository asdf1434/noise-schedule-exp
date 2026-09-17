#!/bin/bash

# ==========================================
# Does InfoNoise reach the FID the best fixed schedule reaches?
#
# This is the whole question, stated plainly. The centre sweeps
# (run_centre_sweep.sh, run_centre_sweep_x01.sh, run_cifar_centre_sweep.sh,
# run_cifar_centre_sweep_scales.sh) measure the best FID any fixed
# logit-normal centre achieves on each of six cells: {mnist, cifar10} x
# {k = 0.1, 1, 10}. InfoNoise claims to match a tuned schedule without being
# tuned. So: run it on the same six cells and compare against the minimum of
# each swept curve. Nothing novel -- a like-for-like check of the paper's claim.
#
# WHAT ALREADY EXISTS (do not re-run; the comparison picks these up from
# master_fid_results.json):
#   mnist        k=1    infonoise                                 10 seeds
#   mnist_x10    k=10   infonoise_sigma_min_0.02_sigma_max_800.0  10 seeds
#   cifar10_x10  k=10   infonoise_sigma_min_0.02_sigma_max_800.0  10 seeds
#
# WHAT THIS SCRIPT ADDS (3 arms x 10 seeds = 30 tasks, ~42 min each):
#   0-9    mnist_x0.1    grid [0.0002, 8]   --t_clip 0.005
#   10-19  cifar10_x0.1  grid [0.0002, 8]   --t_clip 0.005
#   20-29  cifar10       grid default        (t_clip 0.05)
#
# WHY THE 0.1x ARMS ARE RE-RUN. mnist_x0.1 and cifar10_x0.1 already have
# InfoNoise runs on the right grid, but at the default t_clip = 0.05. At k = 0.1
# that puts the loss weight's dead zone at sigma <= 0.0526 = 0.53k, inside the
# informative range -- 51% of the training mass lands in it. The 0.1x centre
# sweeps are run at --t_clip 0.005, so an InfoNoise arm at 0.05 is not
# comparable to them. Same objective or no comparison.
#
# WHY cifar10 k=1 IS NEW. Its only InfoNoise run is gate_c = 0.15 at 5 seeds.
# Every other cell uses the auto gate at 10 seeds; pinning the gate is a
# different method, and 5 seeds is a different power.
#
# GRID BOUNDS scale with the data, k*[0.002, 80], matching the existing x10 and
# x0.1 arms. That is the practice run_gridscale.sh established. Note it does not
# make the gate rule data-determined -- measured gate_c is 7.8-10.4x the grid
# floor on every arm tried, including two different datasets at k = 0.1 that
# return an identical 0.001757 -- so the grid choice still sets the gate. Scaling
# it with the data is simply the choice that puts the gate in the right place.
#
# READING THE RESULT, per cell, paired by seed over the last 5 checkpoints:
#   InfoNoise ~ best swept centre   -> the claim holds here.
#   InfoNoise ~ logit_normal(0,1)   -> no better than the untuned default.
#   InfoNoise << best swept centre  -> the claim fails here; report which cells.
#
# Report a late-epoch average, never best-epoch: InfoNoise runs are less stable
# across checkpoints (per-run epoch-to-epoch SD 8.59 vs 3.87 for the baseline on
# mnist uniform), so a minimum over checkpoints flatters them by selection.
#
#   TRAIN=$(sbatch --parsable scripts/slurm/run_infonoise_vs_optimal.sh)
# then the mnist and cifar eval chains, which now cover all six datasets:
#   P1=$(sbatch --parsable --dependency=afterany:$TRAIN scripts/slurm/run_sweep_eval_prep.sh)
#   E1=$(sbatch --parsable --dependency=afterany:$P1 scripts/slurm/run_sweep_eval.sh)
#   M1=$(sbatch --parsable --dependency=afterany:$E1 scripts/slurm/run_exp1_eval_merge.sh)
#   P2=$(sbatch --parsable --dependency=afterany:$M1 scripts/slurm/run_cifar_sweep_eval_prep.sh)
#   E2=$(sbatch --parsable --dependency=afterany:$P2 scripts/slurm/run_cifar_sweep_eval.sh)
#   sbatch --dependency=afterany:$E2 scripts/slurm/run_exp1_eval_merge.sh
#
# The two eval chains MUST be serialized (P2 waits on M1, not on TRAIN): prep
# reseeds shard files from master, so two chains running at once wipe each
# other's shards. afterany everywhere, never afterok.
# ==========================================
#SBATCH --job-name=infonoise_vs_optimal
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/slurm/slurm_infonoise_vs_optimal_%A_%a.out

set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

NUM_SEEDS=10

# "<dataset>|<dist_params JSON>|<t_clip>"
ARMS=(
    "mnist_x0.1|{\"sigma_min\": 0.0002, \"sigma_max\": 8.0}|0.005"
    "cifar10_x0.1|{\"sigma_min\": 0.0002, \"sigma_max\": 8.0}|0.005"
    "cifar10|{}|0.05"
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
DIST_PARAMS=${REST%|*}
T_CLIP=${REST##*|}

echo "========================================"
echo "Array task $IDX: dataset=$DATASET seed=$SEED t_clip=$T_CLIP"
echo "  dist_params=$DIST_PARAMS"
echo "========================================"

python -u train.py \
    --dataset "$DATASET" \
    --train_dist infonoise \
    --dist_params "$DIST_PARAMS" \
    --t_clip "$T_CLIP" \
    --seed "$SEED"

echo -e "\n========================================"
echo "Array task $IDX ($DATASET/infonoise/seed$SEED) complete."
echo "========================================"
