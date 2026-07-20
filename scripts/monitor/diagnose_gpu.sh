#!/bin/bash

# ==========================================
# One-shot GPU environment diagnostic for the exp3 cuDNN failures
# ("Failed to determine best cudnn convolution algorithm", status 5003,
# reproducing on every node type). Run this INSIDE an interactive
# allocation (see the srun command below) so it reflects the actual GPU
# state train.py sees, then paste the full output back.
#
# Usage (from repo root, on the cluster):
#   srun --account=vision-sitzmann --qos=lab-free \
#     --partition=vision-shared-a6000,vision-shared-a100 \
#     --exclude=isola-2080ti-4,gpu19-2.drl,gpu20-2.drl,improbablex002,isola-ada6000-1,gpu19-1.drl,gpu20-3.drl \
#     --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
#     bash scripts/monitor/diagnose_gpu.sh
# ==========================================

set -x

hostname

echo "----- nvidia-smi -----"
nvidia-smi

echo "----- per-GPU summary -----"
nvidia-smi --query-gpu=driver_version,name,memory.used,memory.total,utilization.gpu --format=csv

echo "----- MIG check -----"
nvidia-smi -q | grep -i mig

echo "----- MPS check -----"
ps aux | grep -i nvidia-cuda-mps | grep -v grep
ls /tmp/nvidia-mps 2>/dev/null && echo "MPS pipe directory exists"

echo "----- jax / jaxlib / cuda versions -----"
source venv/bin/activate
python -c "import jax; print('jax', jax.__version__); print('devices', jax.devices())"
python -c "import jaxlib; print('jaxlib', jaxlib.__version__)"
python -c "import jax; print(jax.print_environment_info())" 2>&1 | head -40

echo "----- actually running one training task now -----"
echo "Watch 'nvidia-smi' from another pane/window while this runs."
python -u train.py --train_dist logit_normal --dist_params '{"mu": 0.0, "sigma": 0.4}' --seed 0
