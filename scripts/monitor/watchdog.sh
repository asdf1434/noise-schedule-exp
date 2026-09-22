#!/bin/bash
# ==========================================
# Requeue array tasks that FAIL, while an experiment is still in flight.
#
# --requeue covers preemption. It does not cover a task that exits nonzero,
# which is what happens when a task lands on a node whose CUDA driver this
# jaxlib build cannot use: train.py's GPU guard catches it and exits 1 within
# two minutes, and the node keeps taking tasks until it is excluded. That cost
# 17 of 40 tasks on isola-2080ti-1 on 2026-09-22 before anyone noticed.
#
# This watches named experiments, requeues their failed tasks, and -- the part
# that matters -- adds any node with repeated failures to the exclude list, so
# the requeued tasks do not land back on it.
#
# Usage:
#   scripts/monitor/watchdog.sh <exp> [<exp> ...]
#   WATCHDOG_INTERVAL=300 WATCHDOG_MAX_RETRIES=2 scripts/monitor/watchdog.sh exp1
#   WATCHDOG_DRY_RUN=1 scripts/monitor/watchdog.sh exp1      # print, submit nothing
#
# Run it detached from a login node:
#   cd <repo> && setsid nohup scripts/monitor/watchdog.sh a b c \
#       > logs/watchdog/run.log 2>&1 < /dev/null &
#
# Exits once every watched experiment has no train task left queued or running.
# ==========================================

set -u

INTERVAL=${WATCHDOG_INTERVAL:-300}
MAX_RETRIES=${WATCHDOG_MAX_RETRIES:-2}
# a node this many failures deep is treated as broken, not unlucky
NODE_STRIKES=${WATCHDOG_NODE_STRIKES:-3}
DRY_RUN=${WATCHDOG_DRY_RUN:-0}

[ "$#" -ge 1 ] || { echo "usage: $0 <experiment> [<experiment> ...]" >&2; exit 1; }

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

STATE_DIR=logs/watchdog
mkdir -p "$STATE_DIR"
RETRIES=$STATE_DIR/retries.txt        # "<exp> <task>" once per requeue
BAD_NODES=$STATE_DIR/bad_nodes.txt    # one node name per line, accumulated
touch "$RETRIES" "$BAD_NODES"

START=$(date +%Y-%m-%dT%H:%M:%S)
log() { echo "[$(date +%H:%M:%S)] $*"; }

EXPERIMENTS=("$@")
log "watching: ${EXPERIMENTS[*]}"
log "interval ${INTERVAL}s, max ${MAX_RETRIES} retries/task, node strike-out at ${NODE_STRIKES}"
[ "$DRY_RUN" = "1" ] && log "DRY RUN -- nothing will be submitted"

# Base --exclude taken from train.sh so the watchdog never widens access to a
# node the scripts already avoid.
base_exclude() {
    grep -m1 -o 'exclude=[^ ]*' scripts/slurm/train.sh | cut -d= -f2
}

exclude_list() {
    local extra
    extra=$(sort -u "$BAD_NODES" | paste -sd, -)
    if [ -n "$extra" ]; then echo "$(base_exclude),$extra"; else base_exclude; fi
}

retries_for() {   # <exp> <task>
    grep -c "^$1 $2\$" "$RETRIES" 2>/dev/null || echo 0
}

while :; do
    still_active=0

    for exp in "${EXPERIMENTS[@]}"; do
        # every generation of this experiment's train job, including requeues
        mapfile -t rows < <(
            sacct -X -S "$START" -n -o JobID,JobName%40,State,NodeList%30 \
                | awk -v n="${exp}_train" '$2 == n {print $1, $3, $4}'
        )

        active=$(squeue -u "$USER" -h -n "${exp}_train" -t PENDING,RUNNING,CONFIGURING,COMPLETING 2>/dev/null | wc -l)
        [ "$active" -gt 0 ] && still_active=1

        # count failures per node, so a genuinely broken node earns its way out
        for row in "${rows[@]:-}"; do
            [ -n "$row" ] || continue
            read -r _jobid _state _node <<< "$row"
            [ "$_state" = "FAILED" ] || continue
            echo "${_node:-unknown}"
        done | sort | uniq -c | while read -r count node; do
            [ "$count" -ge "$NODE_STRIKES" ] || continue
            [ "$node" = "unknown" ] || [ -z "$node" ] && continue
            grep -qx "$node" "$BAD_NODES" && continue
            echo "$node" >> "$BAD_NODES"
            log "NODE STRUCK OUT: $node ($count failures) -- adding to --exclude"
        done

        # collect failed task ids not already requeued to the cap
        to_requeue=()
        for row in "${rows[@]:-}"; do
            [ -n "$row" ] || continue
            read -r _jobid _state _node <<< "$row"
            [ "$_state" = "FAILED" ] || continue
            task=${_jobid##*_}
            case "$task" in ''|*[!0-9]*) continue ;; esac
            n=$(retries_for "$exp" "$task")
            [ "$n" -ge "$MAX_RETRIES" ] && continue
            # skip if a later generation of this task is already back in flight
            squeue -u "$USER" -h -n "${exp}_train" -o "%K" 2>/dev/null \
                | tr ',' '\n' | grep -qx "$task" && continue
            to_requeue+=("$task")
        done

        [ "${#to_requeue[@]}" -eq 0 ] && continue 2>/dev/null || true
        [ "${#to_requeue[@]}" -eq 0 ] && continue

        list=$(printf '%s\n' "${to_requeue[@]}" | sort -un | paste -sd, -)
        log "$exp: requeueing tasks $list (exclude: $(exclude_list))"
        if [ "$DRY_RUN" = "1" ]; then
            log "  [dry run] sbatch --array=$list --exclude=$(exclude_list) ..."
        else
            sbatch --array="$list" \
                   --exclude="$(exclude_list)" \
                   --job-name="${exp}_train" \
                   --output="logs/slurm/slurm_${exp}_train_%A_%a.out" \
                   scripts/slurm/train.sh "$exp" 2>&1 | tail -1
            for t in $(printf '%s\n' "${to_requeue[@]}" | sort -un); do
                echo "$exp $t" >> "$RETRIES"
            done
        fi
        still_active=1
    done

    if [ "$still_active" -eq 0 ]; then
        log "no train tasks left queued or running -- exiting"
        break
    fi
    sleep "$INTERVAL"
done
