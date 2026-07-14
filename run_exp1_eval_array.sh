#!/bin/bash

# ==========================================
# Step 2 of 3 -- scores all eval_runs/ folders in parallel via a Slurm job
# array (8 independent tasks, one process each), the same pattern as
# run_exp1.sh. Each task gets its own log file so you can tail -f a single
# task and see its tqdm progress bars directly, instead of one shared log.
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
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm/slurm_exp1_pilot_eval_array_%A_%a.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

NUM_SHARDS=8

echo "========================================"
echo "Shard $SLURM_ARRAY_TASK_ID / $NUM_SHARDS"
echo "========================================"

python -u evaluate_fid.py --shard "$SLURM_ARRAY_TASK_ID" --num_shards "$NUM_SHARDS"

echo -e "\n========================================"
echo "Shard $SLURM_ARRAY_TASK_ID complete."
echo "Once ALL $NUM_SHARDS array tasks finish, run merge_fid_shards.py -- see"
echo "run_exp1_eval_merge.sh (submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
