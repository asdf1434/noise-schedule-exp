#!/bin/bash

# ==========================================
# Stage 3 of the FID chain: fold results/fid_shards/*.json back into
# results/master_fid_results.json.
#
# Not experiment-specific -- merge_fid_shards.py takes whatever shard files are
# present -- but it runs per chain so submit.sh can hang it off the eval array
# with --dependency. Note the dependency submit.sh uses is afterany, not
# afterok: a single preempted-and-not-retried shard should not strand the
# results that every other shard already computed.
#
# Submitted by submit.sh; not usually run by hand.
# ==========================================
#SBATCH --job-name=merge
#SBATCH --account=vision-sitzmann
#SBATCH --qos=lab-free
#SBATCH --requeue
# A partition list is REQUIRED: with qos=lab-free and no partition, sbatch
# rejects the job outright ("partition nil, qos lab-free / Invalid qos
# specification"). The per-experiment launchers this replaced all carried one;
# it was dropped in the consolidation, which broke the merge stage everywhere.
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --exclude=isola-v100-2,isola-2080ti-1,andreas-h100-1,isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,gpu19-1.drl,isola-ada6000-1,gpu20-3.drl,freeman-titanrtx-2,isola-3080-1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/slurm_merge_%j.out

set -e

# Under Slurm, BASH_SOURCE[0] is the node-local copy of the batch script
# (/var/lib/slurm/slurmd/job*/slurm_script), NOT this file in the repo, so it
# cannot locate the repo. submit.sh cd's to the repo root before calling
# sbatch, which is what makes SLURM_SUBMIT_DIR correct here.
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"

echo "+ merge_fid_shards.py"
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

mkdir -p logs/slurm
source venv/bin/activate
python -u merge_fid_shards.py
