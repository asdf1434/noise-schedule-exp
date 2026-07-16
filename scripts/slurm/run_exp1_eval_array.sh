#!/bin/bash

# ==========================================
# Step 2 of 3 -- scores all eval_runs/ folders in parallel via a Slurm job
# array (64 independent tasks, one process each, each on its own GPU), the
# same pattern as run_exp1.sh. Each task gets its own log file so you can
# tail -f a single task and see its tqdm progress bars directly, instead of
# one shared log.
#
# Shard count: with GPU + a resident InceptionV3 feature extractor (built
# once per process, not once per folder), per-folder cost is now seconds
# rather than the ~3 min/folder CPU baseline this was originally sized for.
# That argues for MORE parallelism, not less -- wall-clock is (folders per
# shard) x (time per folder), so fewer shards just means more sequential
# work per shard. Kept at 64 (same as the CPU version) as a level that's
# already known to schedule reasonably under the lab-free QOS; if the QOS
# grants that many concurrent GPU allocations without heavy queueing, it's
# worth experimenting with pushing this higher since per-job overhead
# (venv activation, one-time model build) is now small relative to the
# potential wall-clock win.
#
# Run AFTER run_exp1_eval.sh (real-image stats must already be cached):
#   sbatch --dependency=afterok:<STATS_JOBID> scripts/slurm/run_exp1_eval_array.sh
#
# Each task writes its own master_fid_results_shard<N>.json -- run
# run_exp1_eval_merge.sh afterward to combine them into master_fid_results.json.
# ==========================================
#SBATCH --job-name=exp1_pilot_eval_array
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --array=0-63
#SBATCH --output=logs/slurm/slurm_exp1_pilot_eval_array_%A_%a.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

NUM_SHARDS=64

echo "========================================"
echo "Shard $SLURM_ARRAY_TASK_ID / $NUM_SHARDS"
echo "========================================"

python -u evaluate_fid.py --shard "$SLURM_ARRAY_TASK_ID" --num_shards "$NUM_SHARDS"

echo -e "\n========================================"
echo "Shard $SLURM_ARRAY_TASK_ID complete."
echo "Once ALL $NUM_SHARDS array tasks finish, run merge_fid_shards.py -- see"
echo "scripts/slurm/run_exp1_eval_merge.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
