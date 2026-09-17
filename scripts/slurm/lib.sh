#!/bin/bash

# Shared by train.sh / eval_prep.sh / eval.sh / merge.sh / submit.sh.
# Sourced, never executed.

EXPERIMENTS_DIR="scripts/slurm/experiments"

# load_experiment <name>
#
# Sources scripts/slurm/experiments/<name>.conf and checks that it defines what
# the job scripts need. Defaults fill in the values most experiments share, so a
# .conf only states what is unusual about it.
load_experiment() {
    EXPERIMENT="$1"
    local conf="$EXPERIMENTS_DIR/$EXPERIMENT.conf"

    if [ ! -f "$conf" ]; then
        echo "Unknown experiment '$EXPERIMENT' (no $conf)." >&2
        echo "Available:" >&2
        ls "$EXPERIMENTS_DIR" 2>/dev/null | sed 's/\.conf$//' | sed 's/^/  /' >&2
        return 1
    fi

    # Defaults, overridable by the .conf.
    SEEDS=
    TIME=03:00:00
    MEM=32G
    CPUS=4
    EVAL_DATASETS=()
    EVAL_SHARDS=16
    EVAL_TIME=03:30:00
    GENERATE_REAL=()
    CONFIGS=()

    # shellcheck disable=SC1090
    source "$conf"

    local missing=()
    [ -n "$SEEDS" ] || missing+=(SEEDS)
    [ "${#CONFIGS[@]}" -gt 0 ] || missing+=(CONFIGS)
    [ "${#EVAL_DATASETS[@]}" -gt 0 ] || missing+=(EVAL_DATASETS)
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "$conf does not define: ${missing[*]}" >&2
        return 1
    fi

    NUM_TASKS=$(( ${#CONFIGS[@]} * SEEDS ))
}
