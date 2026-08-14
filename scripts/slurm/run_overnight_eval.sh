#!/bin/bash

# ==========================================
# Step 3 of 3 -- score every new dataset's eval_runs/ folders.
#
# Covers both experiments launching tonight, six datasets in one array:
#   fashion_mnist_r14 / _r28 / _r56    (run_fashion_sizes.sh)
#   mnist_g050 / g100 / g250           (run_mnist_curves.sh)
#
# Work per dataset: 30 runs x 10 epochs x 9 step spacings = 2,700 scores,
# so 16,200 in total. Each dataset gets 16 shards.
#
#   dataset_idx = ID / 16,  shard = ID % 16
#
# Shard files are named per (shard, dataset), so all 96 tasks write to
# different files and never race.
#
# BEFORE SUBMITTING -- seed the shard files, or the merge will delete results:
#   for d in fashion_mnist_r14 fashion_mnist_r28 fashion_mnist_r56 \
#            mnist_g050 mnist_g100 mnist_g250; do
#     python scripts/monitor/seed_fid_shards.py --dataset $d --num_shards 16
#   done
#
# merge_fid_shards.py rebuilds master_fid_results.json purely from the shard
# files it can find, so any shard that hasn't been primed with the current
# master contributes nothing and the merged file comes out missing everything
# those shards didn't personally score. Seeding first also means each shard
# skips what's already scored instead of redoing another shard's work.
#
# Submit after both training arrays and the prep job finish:
#   sbatch --dependency=afterok:<TRAIN1>:<TRAIN2>:<PREP> scripts/slurm/run_overnight_eval.sh
#   sbatch --dependency=afterok:<THIS_JOBID> scripts/slurm/run_exp1_eval_merge.sh
# ==========================================
#SBATCH --job-name=overnight_eval
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --array=0-95
#SBATCH --output=logs/slurm/slurm_overnight_eval_%A_%a.out

set -e

mkdir -p logs/slurm

source venv/bin/activate

NUM_SHARDS=16
DATASETS=(fashion_mnist_r14 fashion_mnist_r28 fashion_mnist_r56 mnist_g050 mnist_g100 mnist_g250)

IDX=$SLURM_ARRAY_TASK_ID
DS_IDX=$((IDX / NUM_SHARDS))
SHARD=$((IDX % NUM_SHARDS))
DATASET=${DATASETS[$DS_IDX]}

echo "========================================"
echo "Task $IDX: dataset=$DATASET shard $SHARD / $NUM_SHARDS"
echo "========================================"

python -u evaluate_fid.py --shard "$SHARD" --num_shards "$NUM_SHARDS" --dataset "$DATASET"

echo -e "\n========================================"
echo "Task $IDX ($DATASET shard $SHARD) complete."
echo "Once ALL 96 tasks finish, merge with scripts/slurm/run_exp1_eval_merge.sh"
echo "(submit with --dependency=afterok:<this array job's ID>)."
echo "========================================"
