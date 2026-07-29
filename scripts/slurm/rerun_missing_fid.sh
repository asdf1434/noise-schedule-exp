#!/bin/bash

# ==========================================
# Re-score every eval_runs/ folder that has samples but no FID number, for one
# dataset, without redoing the ones already in results/master_fid_results.json.
#
# Why this exists: evaluate_fid.py resumes from its OWN shard file, not from the
# merged master, so naively resubmitting an eval array after some shards hit the
# time limit makes each surviving shard treat every folder it didn't personally
# score as pending. Seeding all the shard files with the merged master first
# makes pending_dirs exactly the unscored set, which the stride slice then
# partitions cleanly across tasks. (It's also what keeps the existing numbers
# alive through merge_fid_shards.py, which rebuilds master purely from shards.)
#
# One pass covers every conditioning variant of the dataset -- evaluate_fid.py
# filters by dataset only -- and picks up pre-rename exp1/exp2/eurosat dirs too.
#
# Check what's missing first:
#   python scripts/monitor/check_missing.py --dataset cifar10 --conditioning inpaint
#
# Usage: scripts/slurm/rerun_missing_fid.sh <dataset> [num_shards]
# Example: scripts/slurm/rerun_missing_fid.sh cifar10
# ==========================================

set -e

DATASET=$1
NUM_SHARDS=${2:-64}

if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset> [num_shards]" >&2
    echo "Datasets: mnist cifar10 fashion_mnist eurosat64 eurosat" >&2
    exit 1
fi

# mnist's eval scripts still carry their exp1 names (they predate the rename).
case "$DATASET" in
    mnist)         EVAL_ARRAY=run_exp1_eval_array.sh;      EVAL_MERGE=run_exp1_eval_merge.sh ;;
    cifar10)       EVAL_ARRAY=run_cifar10_eval_array.sh;   EVAL_MERGE=run_cifar10_eval_merge.sh ;;
    fashion_mnist) EVAL_ARRAY=run_fashionmnist_eval_array.sh; EVAL_MERGE=run_fashionmnist_eval_merge.sh ;;
    eurosat64)     EVAL_ARRAY=run_eurosat64_eval_array.sh; EVAL_MERGE=run_eurosat64_eval_merge.sh ;;
    eurosat)       EVAL_ARRAY=run_eurosat_eval_array.sh;   EVAL_MERGE=run_eurosat_eval_merge.sh ;;
    *)
        echo "Unknown dataset '$DATASET'." >&2
        exit 1
        ;;
esac

mkdir -p logs/slurm

source venv/bin/activate

python scripts/monitor/seed_fid_shards.py --dataset "$DATASET" --num_shards "$NUM_SHARDS"

EVAL_JOBID=$(sbatch --parsable "scripts/slurm/$EVAL_ARRAY")
echo "1/2 FID eval array: job $EVAL_JOBID  ($EVAL_ARRAY)"

MERGE_JOBID=$(sbatch --parsable --dependency=afterok:"$EVAL_JOBID" "scripts/slurm/$EVAL_MERGE")
echo "2/2 merge:          job $MERGE_JOBID  (after $EVAL_JOBID, $EVAL_MERGE)"

echo ""
echo "Track with: squeue -u \$USER"
echo "After the merge lands, confirm the holes are gone:"
echo "  python scripts/monitor/check_missing.py --dataset $DATASET --conditioning none"
echo "If tasks fail: python scripts/monitor/check_slurm_logs.py \"*${EVAL_JOBID}*\""
