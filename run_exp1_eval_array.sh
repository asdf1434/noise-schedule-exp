#!/bin/bash

# ==========================================
# Step 2 of 3 -- scores all eval_runs/ folders in parallel via a Slurm job
# array (64 independent tasks, one process each), the same pattern as
# run_exp1.sh. Each task gets its own log file so you can tail -f a single
# task and see its tqdm progress bars directly, instead of one shared log.
#
# Shard count sized for exp1 + exp2 combined: exp1's 400 folders are already
# scored (evaluate_fid.py's resume logic skips them), so this is really
# sized for exp2's ~3200 pending folders (80 experiments x 10 epochs x 4
# schedules) at ~3 min/folder => ~9600 total folder-minutes. At 8 shards
# that's ~20h/shard (way past the time limit below); at 64 shards it's
# ~2.5h/shard, comfortably inside it.
#
# Run AFTER run_exp1_eval.sh (real-image stats must already be cached):
#   sbatch --dependency=afterok:<STATS_JOBID> run_exp1_eval_array.sh
#
# Each task writes its own master_fid_results_shard<N>.json -- run
# run_exp1_eval_merge.sh afterward to combine them into master_fid_results.json.
# ==========================================
#SBATCH --job-name=exp1_pilot_eval_array
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --cpus-per-task=4
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
echo "run_exp1_eval_merge.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
