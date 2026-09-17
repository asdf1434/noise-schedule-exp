#!/bin/bash

# ==========================================
# Submit one experiment's full train -> FID chain.
#
#     scripts/slurm/submit.sh <experiment> [stage ...]
#     scripts/slurm/submit.sh --list
#     scripts/slurm/submit.sh --dry-run <experiment> [stage ...]
#
# Experiments are the files in scripts/slurm/experiments/. Each one names its
# training grid, its walltime, and the datasets its FID stage scores; this
# script reads that and submits four jobs, chained so each waits for the last:
#
#     train -> eval_prep -> eval -> merge
#
# The #SBATCH directives inside train.sh and eval.sh are placeholders. The
# per-experiment values (job name, walltime, array size, log path) are passed on
# the sbatch command line, where they override the in-file directives -- which
# is the only way to do it, since sbatch reads those directives before the shell
# ever runs and so cannot see a computed array size.
#
# Stages are chained with --dependency=afterany rather than afterok on purpose.
# lab-free is preemptible: with afterok, one task that is preempted and not
# requeued leaves the rest of the chain in DependencyNeverSatisfied forever,
# which is the failure that silently truncated several sweeps. afterany runs the
# next stage on whatever finished, and check_missing.py finds the gaps.
#
# Pass explicit stages to run part of the chain, e.g. to rescore without
# retraining:
#
#     scripts/slurm/submit.sh mnist_inpaint eval_prep eval merge
# ==========================================

set -e

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

source scripts/slurm/lib.sh

DRY_RUN=0
if [ "$1" = "--dry-run" ] || [ "$1" = "-n" ]; then
    DRY_RUN=1
    shift
fi

if [ "$1" = "--list" ] || [ -z "$1" ]; then
    echo "Experiments (scripts/slurm/experiments/):"
    echo
    printf "  %-28s %8s %6s %8s  %s\n" EXPERIMENT CONFIGS SEEDS TASKS DATASETS
    for conf in "$EXPERIMENTS_DIR"/*.conf; do
        name=$(basename "$conf" .conf)
        ( load_experiment "$name" >/dev/null 2>&1 &&
          printf "  %-28s %8d %6d %8d  %s\n" \
                 "$name" "${#CONFIGS[@]}" "$SEEDS" "$NUM_TASKS" "${EVAL_DATASETS[*]}" )
    done
    echo
    echo "Usage: scripts/slurm/submit.sh [--dry-run] <experiment> [train|eval_prep|eval|merge ...]"
    [ -z "$1" ] && exit 1
    exit 0
fi

EXPERIMENT=$1
shift
load_experiment "$EXPERIMENT"

STAGES=("$@")
[ "${#STAGES[@]}" -gt 0 ] || STAGES=(train eval_prep eval merge)

echo "Experiment : $EXPERIMENT"
echo "Training   : ${#CONFIGS[@]} configs x $SEEDS seeds = $NUM_TASKS tasks, $TIME each"
echo "Scoring    : ${EVAL_DATASETS[*]} across $EVAL_SHARDS shards"
echo "Stages     : ${STAGES[*]}"
echo

# submit <script> <dependency-or-empty> <sbatch options...>
# Echoes the command, runs it unless --dry-run, and prints the new job id.
submit() {
    local script=$1 dep=$2
    shift 2
    # No --export: --export=ALL,VAR=val makes this cluster attempt user-env
    # retrieval at launch, which fails and leaves every task held. The
    # experiment name goes to the script as a positional argument instead.
    local args=(--parsable)
    [ -n "$dep" ] && args+=(--dependency=afterany:"$dep")
    args+=("$@" "scripts/slurm/$script" "$EXPERIMENT")

    echo "+ sbatch ${args[*]}" >&2
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRYRUN_$script"
    else
        sbatch "${args[@]}"
    fi
}

DEP=""
for stage in "${STAGES[@]}"; do
    case "$stage" in
        train)
            DEP=$(submit train.sh "$DEP" \
                --job-name="${EXPERIMENT}_train" \
                --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS" \
                --array="0-$((NUM_TASKS - 1))" \
                --output="logs/slurm/slurm_${EXPERIMENT}_train_%A_%a.out")
            echo "  train      job $DEP  (array 0-$((NUM_TASKS - 1)))"
            ;;
        eval_prep)
            DEP=$(submit eval_prep.sh "$DEP" \
                --job-name="${EXPERIMENT}_eval_prep" \
                --output="logs/slurm/slurm_${EXPERIMENT}_eval_prep_%j.out")
            echo "  eval_prep  job $DEP"
            ;;
        eval)
            DEP=$(submit eval.sh "$DEP" \
                --job-name="${EXPERIMENT}_eval" \
                --time="$EVAL_TIME" \
                --array="0-$((EVAL_SHARDS - 1))" \
                --output="logs/slurm/slurm_${EXPERIMENT}_eval_%A_%a.out")
            echo "  eval       job $DEP  (array 0-$((EVAL_SHARDS - 1)))"
            ;;
        merge)
            DEP=$(submit merge.sh "$DEP" \
                --job-name="${EXPERIMENT}_merge" \
                --output="logs/slurm/slurm_${EXPERIMENT}_merge_%j.out")
            echo "  merge      job $DEP"
            ;;
        *)
            echo "Unknown stage '$stage' (expected train, eval_prep, eval or merge)." >&2
            exit 1
            ;;
    esac
done

echo
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run -- nothing was submitted."
else
    echo "Submitted. Watch with: squeue -u \$USER"
    echo "Check for gaps afterwards with: python scripts/monitor/check_missing.py --dataset ${EVAL_DATASETS[0]}"
fi
