#!/bin/bash

# ==========================================
# Stage 1 of 3 -- scoring prep for the mnist_x10 shifted-regime run.
#
# mnist_x10 shares mnist's real images and cached stats (real_dir=data/real,
# real_stats_name=mnist_real), so the cache_real_stats call below is normally a
# no-op. It is kept because it is idempotent and because a missing cache makes
# evaluate_fid.py exit rather than score anything.
#
# The seeding step is the one that matters. merge_fid_shards.py rebuilds
# master_fid_results.json purely from the shard files it finds, and
# evaluate_fid.py resumes from its OWN shard file only -- so without seeding,
# the merge silently DROPS every previously scored cell that no shard happened
# to rescore. There are no mnist_x10 shard files yet, but master already holds
# every other dataset's results, and those are what would be lost.
#
# NUM_SHARDS must match run_mnist_x10_eval.sh.
#
# Submit as stage 1 of the chain:
#   PREP=$(sbatch --parsable scripts/slurm/run_mnist_x10_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_mnist_x10_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=mnist_x10_eval_prep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm_mnist_x10_eval_prep_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

# Keep in sync with run_mnist_x10_eval.sh.
DATASET=mnist_x10
NUM_SHARDS=16

echo "========================================"
echo "Caching real-image FID stats for $DATASET (shares mnist's; expected no-op)"
echo "========================================"
python -u cache_real_stats.py --dataset "$DATASET"

echo -e "\n========================================"
echo "Seeding $NUM_SHARDS shard files from results/master_fid_results.json"
echo "========================================"
if [ -f results/master_fid_results.json ]; then
    python -u scripts/monitor/seed_fid_shards.py \
        --dataset "$DATASET" --num_shards "$NUM_SHARDS"
else
    echo "No master results yet -- nothing to preserve, starting from empty shards."
fi

echo -e "\n========================================"
echo "Prep complete."
echo "========================================"
