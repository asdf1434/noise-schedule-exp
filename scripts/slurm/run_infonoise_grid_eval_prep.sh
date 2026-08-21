#!/bin/bash

# ==========================================
# Stage 1 of 3 -- scoring prep for the InfoNoise generalization grid.
#
# Two jobs, both of which MUST happen before the eval array runs:
#
#   1. Cache each dataset's real-image Inception stats. Every dataset is
#      scored against its OWN folder of real images, so a missing cache means
#      that dataset simply can't be scored.
#
#   2. Seed the per-shard result files from the current master. This is the
#      step whose absence has already cost this project a sweep:
#      merge_fid_shards.py rebuilds master_fid_results.json purely from the
#      shard files it finds, and evaluate_fid.py resumes from its OWN shard
#      file only. Without seeding, each shard treats every folder it didn't
#      personally score as pending (duplicated work), and the merge silently
#      DROPS every previously scored cell that no shard happened to rescore.
#
# Runs sequentially in one job -- minutes of work, not worth an array.
# Safe to re-run: cache_real_stats.py skips anything already cached, and
# seeding only ever copies master into the shard files.
#
# NUM_SHARDS must match run_infonoise_grid_eval.sh's shards-per-dataset, or
# the array will write files the merge never sees seeded (and vice versa).
#
# Note on stale shard files: earlier sweeps seeded mnist at 64 shards, so
# results/fid_shards/ still holds master_fid_results_shard{16..63}_mnist.json.
# Those are left alone and still merged, which is harmless -- FID for a given
# (experiment, schedule, epoch) is deterministic, so the union can't conflict.
#
# Submit as stage 1 of the chain:
#   PREP=$(sbatch --parsable scripts/slurm/run_infonoise_grid_eval_prep.sh)
#   EVAL=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm/run_infonoise_grid_eval.sh)
#   sbatch --dependency=afterok:$EVAL scripts/slurm/run_exp1_eval_merge.sh
#
# The merge stage is reused as-is: merge_fid_shards.py globs every shard file
# regardless of dataset, so it needs no grid-specific version.
# ==========================================
#SBATCH --job-name=infonoise_grid_eval_prep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/slurm_infonoise_grid_eval_prep_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

# Keep in sync with run_infonoise_grid_eval.sh.
DATASETS=(mnist fashion_mnist cifar10 eurosat64)
NUM_SHARDS=16

for DS in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Caching real-image FID stats for $DS"
    echo "========================================"
    python -u cache_real_stats.py --dataset "$DS"
done

echo -e "\n========================================"
echo "Seeding $NUM_SHARDS shard files per dataset from results/master_fid_results.json"
echo "========================================"

# On a fresh checkout there is no master yet and seed_fid_shards.py would fail
# opening it. Nothing to preserve in that case, so skip rather than abort the
# chain -- starting from empty shards is correct when there are no earlier
# results to drop.
if [ -f results/master_fid_results.json ]; then
    for DS in "${DATASETS[@]}"; do
        python -u scripts/monitor/seed_fid_shards.py \
            --dataset "$DS" --num_shards "$NUM_SHARDS"
    done
else
    echo "No results/master_fid_results.json yet -- nothing to seed, skipping."
fi

echo -e "\n========================================"
echo "Prep complete for: ${DATASETS[*]}"
echo "Next: sbatch --dependency=afterok:$SLURM_JOB_ID scripts/slurm/run_infonoise_grid_eval.sh"
echo "========================================"
