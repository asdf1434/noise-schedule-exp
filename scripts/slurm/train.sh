#!/bin/bash

# ==========================================
# Generic training array job.
#
# Runs one (config, seed) cell of an experiment defined in
# scripts/slurm/experiments/<EXPERIMENT>.conf. The experiment name arrives in
# the EXPERIMENT environment variable, and the array index selects the cell:
#
#     config_index = SLURM_ARRAY_TASK_ID / SEEDS
#     seed         = SLURM_ARRAY_TASK_ID % SEEDS
#
# That is the same index mapping every hand-written launcher used, so array
# indices printed by scripts/monitor/check_missing.py still mean what they did.
#
# Do not sbatch this directly -- the #SBATCH values below are placeholders, and
# submit.sh overrides the ones that vary per experiment (job name, walltime,
# array size, output path) on the sbatch command line, where they take
# precedence over these directives. Use:
#
#     scripts/slurm/submit.sh <experiment>
#
# See the canonical header notes in CLAUDE.md before editing the partition or
# exclude lists; they are identical across every GPU job in this repo on
# purpose, and the omission of vision-shared-v100 is deliberate.
# ==========================================
#SBATCH --job-name=train
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/slurm_train_%A_%a.out

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

IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set -- this script runs as a job array}
CFG_IDX=$((IDX / SEEDS))
SEED=$((IDX % SEEDS))

if [ "$CFG_IDX" -ge "${#CONFIGS[@]}" ]; then
    echo "Array index $IDX is past the end of $EXPERIMENT: ${#CONFIGS[@]} configs x $SEEDS seeds = $(( ${#CONFIGS[@]} * SEEDS )) tasks." >&2
    exit 1
fi

# Unquoted on purpose: the config is a space-separated word list, and any JSON
# in it is written without spaces so it survives as a single word.
set -- ${CONFIGS[$CFG_IDX]}

echo "========================================"
echo "$EXPERIMENT task $IDX: config $CFG_IDX of ${#CONFIGS[@]}, seed $SEED"
echo "  train.py $* --seed $SEED"
echo "========================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

mkdir -p logs/slurm logs/metrics
source venv/bin/activate

python -u train.py "$@" --seed "$SEED"

echo -e "\n========================================"
echo "$EXPERIMENT task $IDX complete (config $CFG_IDX, seed $SEED)."
echo "========================================"
