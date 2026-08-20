#!/bin/bash

# ==========================================
# Stage 2 of the InfoNoise overnight chain: cache real-image FID stats AND
# seed the per-shard result files before the eval array runs.
#
# This exists instead of run_exp1_eval.sh because of a failure mode that has
# already cost this project a sweep. merge_fid_shards.py rebuilds
# master_fid_results.json purely from the shard files it finds, and
# evaluate_fid.py resumes from its OWN shard file only. So if the shards
# aren't primed with the current master first:
#
#   - every shard treats folders it didn't personally score as pending and
#     redoes another shard's work, and
#   - the merge silently DROPS every previously scored cell that no shard
#     happened to rescore.
#
# Seeding first makes the pending set exactly the folders nobody has scored
# yet, so this run adds the 40 new InfoNoise runs' cells to the existing
# results rather than replacing them.
#
# Both steps are safe to re-run: cache_real_stats.py skips anything already
# cached, and seeding only ever copies master into the shard files.
#
# Submit as stage 2 of the chain (this is what run_pipeline.sh does):
#   scripts/slurm/run_pipeline.sh scripts/slurm/run_infonoise.sh \
#       scripts/slurm/run_infonoise_eval_prep.sh
#
# NUM_SHARDS must match the --array width of the eval array that follows.
# run_exp1_eval_array.sh uses 64 shards (--array=0-63); change both together
# or the merge will be missing whatever the extra shards would have scored.
# ==========================================
#SBATCH --job-name=infonoise_eval_prep
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/slurm_infonoise_eval_prep_%j.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

DATASET=mnist
NUM_SHARDS=64

echo "========================================"
echo "Caching real-image FID stats for $DATASET"
echo "========================================"

python -u cache_real_stats.py --dataset "$DATASET"

echo -e "\n========================================"
echo "Seeding $NUM_SHARDS shard files from results/master_fid_results.json"
echo "========================================"

# On a fresh checkout there is no master yet, and seed_fid_shards.py would
# fail opening it. Nothing to preserve in that case, so skip rather than
# abort the chain -- the eval array starts from empty shards, which is
# correct when there are no earlier results to drop.
if [ -f results/master_fid_results.json ]; then
    python -u scripts/monitor/seed_fid_shards.py \
        --dataset "$DATASET" --num_shards "$NUM_SHARDS"
else
    echo "No results/master_fid_results.json yet -- nothing to seed, skipping."
fi

echo -e "\n========================================"
echo "Prep done. Next: sbatch --dependency=afterok:$SLURM_JOB_ID scripts/slurm/run_exp1_eval_array.sh"
echo "========================================"
