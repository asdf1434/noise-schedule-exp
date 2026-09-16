#!/bin/bash

# ==========================================
# Stage 1 of 3 -- scoring prep for the training-distribution centre sweep.
#
# Supersedes run_mnist_x10_eval_prep.sh, which covered mnist_x10 alone.
#
# Each scaled dataset shares its unscaled parent's real images and cached stats
# (mnist_x* -> data/real / mnist_real, cifar10_x* -> data/real_cifar10 /
# cifar10_real), so cache_real_stats is normally a no-op here. It stays because
# it is idempotent and a missing cache makes evaluate_fid.py exit without
# scoring anything.
#
# The seeding step is the one that matters: merge_fid_shards.py rebuilds
# master_fid_results.json purely from the shard files it finds, so any cell no
# shard has is silently dropped from the merged result.
#
# NUM_SHARDS must match run_shift_eval.sh.
# ==========================================
#SBATCH --job-name=sweep_eval_prep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm_sweep_eval_prep_%j.out

set -e
mkdir -p logs/slurm
source venv/bin/activate

# Keep in sync with run_sweep_eval.sh.
DATASETS=(mnist mnist_x10 mnist_x0.1)
NUM_SHARDS=16

for DS in "${DATASETS[@]}"; do
    echo "=== caching real-image FID stats for $DS (shares its unscaled parent's) ==="
    python -u cache_real_stats.py --dataset "$DS"
done

if [ -f results/master_fid_results.json ]; then
    for DS in "${DATASETS[@]}"; do
        python -u scripts/monitor/seed_fid_shards.py --dataset "$DS" --num_shards "$NUM_SHARDS"
    done
else
    echo "No master results yet -- starting from empty shards."
fi

echo "Prep complete."
