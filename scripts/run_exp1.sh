#!/bin/bash

# ==========================================
# 1. Slurm Resource Requests
# ==========================================
# Job array: 10 independent tasks (SLURM_ARRAY_TASK_ID 0..9), one per
# (train_dist, seed) combo, running in PARALLEL across GPUs instead of
# sequentially in one long job. Each task only does ONE training run, so
# the per-task time limit only needs to cover a single ~100-epoch run
# (measured ~25s/epoch => ~42 min), not all 10.
#SBATCH --job-name=exp1_pilot
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=logs/slurm/slurm_exp1_pilot_%A_%a.out

# Exit immediately if a command exits with a non-zero status
set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

# ==========================================
# 2. Map array task ID -> (train_dist, seed)
# ==========================================
# index:  0        1        2        3        4        5              6              7              8              9
# dist:   uniform  uniform  uniform  uniform  uniform  logit_normal   logit_normal   logit_normal   logit_normal   logit_normal
# seed:   0        1        2        3        4        0              1              2              3              4
DISTS=(uniform uniform uniform uniform uniform logit_normal logit_normal logit_normal logit_normal logit_normal)
SEEDS=(0 1 2 3 4 0 1 2 3 4)

IDX=$SLURM_ARRAY_TASK_ID
DIST=${DISTS[$IDX]}
SEED=${SEEDS[$IDX]}

echo "========================================"
echo "Array task $IDX: train_dist=$DIST seed=$SEED"
echo "========================================"

if [ "$DIST" == "uniform" ]; then
    python -u train.py --train_dist uniform --seed "$SEED"
else
    python -u train.py \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
fi

echo -e "\n========================================"
echo "Array task $IDX ($DIST seed $SEED) complete."
echo "Once ALL 10 array tasks finish, run evaluate_fid.py separately -- see"
echo "run_exp1_eval.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
