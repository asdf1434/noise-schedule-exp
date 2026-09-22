#!/bin/bash

# ==========================================
# Stage 2 of the FID chain: score this experiment's eval_runs/ folders.
#
# One array task per shard; each task scores its shard of every dataset the
# experiment covers, writing results/fid_shards/<...>_shard<N>.json. The merge
# stage combines them afterwards.
#
# EVAL_SHARDS comes from the experiment's .conf and sets the array size, so
# raising it splits the same work across more, shorter tasks. Keep an eye on it
# against the 03:30:00 walltime when adding datasets to an experiment.
#
# The partition list deliberately drops vision-shared-rtx2080ti and
# vision-shared-titanrtx: clean-fid's InceptionV3 does not fit comfortably on
# those cards. Do not unify it with train.sh's list.
#
# Submitted by submit.sh; not usually run by hand.
# ==========================================
#SBATCH --job-name=eval
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,isola-2080ti-1,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --output=logs/slurm/slurm_eval_%A_%a.out

set -e

# Under Slurm, BASH_SOURCE[0] is the node-local copy of the batch script
# (/var/lib/slurm/slurmd/job*/slurm_script), NOT this file in the repo, so it
# cannot locate the repo. submit.sh cd's to the repo root before calling
# sbatch, which is what makes SLURM_SUBMIT_DIR correct here. The BASH_SOURCE
# fallback is for running this script directly (e.g. with DRY_RUN=1).
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"

source scripts/slurm/lib.sh
# The experiment name arrives as the first positional argument. It used to be
# passed as --export=ALL,EXPERIMENT=..., which put every task into
# user_env_retrieval_failed_requeued_held on this cluster -- Slurm tried to
# retrieve the user environment at launch, failed, requeued and held the job,
# so it never ran and never reported an error. The env var still works as a
# fallback for anything that sets it directly.
EXPERIMENT=${1:-${EXPERIMENT:-}}
load_experiment "${EXPERIMENT:?EXPERIMENT is not set -- submit with scripts/slurm/submit.sh <experiment>}"

SHARD=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set -- this script runs as a job array}

echo "========================================"
echo "$EXPERIMENT eval: shard $SHARD of $EVAL_SHARDS, datasets ${EVAL_DATASETS[*]}"
echo "========================================"

if [ "${DRY_RUN:-0}" != "1" ]; then
    mkdir -p logs/slurm results/fid_shards
    source venv/bin/activate
fi

for DS in "${EVAL_DATASETS[@]}"; do
    echo "+ evaluate_fid.py --shard $SHARD --num_shards $EVAL_SHARDS --dataset $DS"
    [ "${DRY_RUN:-0}" = "1" ] || \
        python -u evaluate_fid.py --shard "$SHARD" --num_shards "$EVAL_SHARDS" --dataset "$DS"
done

echo "$EXPERIMENT eval shard $SHARD complete."
