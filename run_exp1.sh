#!/bin/bash

# ==========================================
# 1. Slurm Resource Requests
# ==========================================
#SBATCH --job-name=exp1_pilot           
#SBATCH --account=vision-sitzmann     
#SBATCH --qos=lab-free
#SBATCH --partition=vision-shared-rtx2080ti,vision-shared-titanrtx,vision-shared-a6000,vision-shared-a100,vision-shared-l40s,vision-shared-h100,vision-shared-h200,vision-shared-v100,vision-shared-rtx3090,vision-shared-rtx3080,vision-shared-rtx6000ada,vision-shared-rtx4090,csail-shared-h200,csail-shared-l40s
#SBATCH --gres=gpu:1        
#SBATCH --cpus-per-task=4             
#SBATCH --mem=32G                     
#SBATCH --time=05:00:00       
#SBATCH --output=logs/slurm/slurm_exp1_pilot_%j.out 

# Exit immediately if a command exits with a non-zero status
set -e

mkdir -p logs/slurm
mkdir -p logs/metrics

source venv/bin/activate

echo "========================================"
echo "does training-time noise distribution / inference-time sampling actually affect FID"
echo ""
echo "2 training distributions (uniform, logit_normal mu=0.0 sigma=1.0) x 5 seeds = 10 runs"
echo "each run scored against all 4 sampling schedules automatically"
echo "========================================"

for SEED in 0 1 2 3 4; do
    echo -e "\n>>> [seed $SEED] Uniform baseline"
    python -u train.py --train_dist uniform --seed "$SEED"

    echo -e "\n>>> [seed $SEED] Logit-Normal (mu=0.0, sigma=1.0)"
    python -u train.py \
        --train_dist logit_normal \
        --dist_params '{"mu": 0.0, "sigma": 1.0}' \
        --seed "$SEED"
done

echo -e "\n========================================"
echo "All 10 pilot runs complete! Starting offline evaluation phase..."
echo "========================================"

# Run the evaluation script to generate master_fid_results.json
python evaluate_fid.py

echo -e "\n========================================"
echo "Pilot done. master_fid_results.json now has 10 experiment entries:"
echo "  uniform_seed0 .. uniform_seed4"
echo "  logit_normal_mu_0.0_sigma_1.0_seed0 .. _seed4"
echo "each scored against all 4 sampling schedules per eval epoch."
echo ""
echo "NOTE: plot.py only knows how to plot a single named experiment -- it does"
echo "NOT yet aggregate mean/std across the 5 seeds per (train_dist, schedule)"
echo "cell. That aggregation + comparison step still needs to be written before"
echo "drawing conclusions from this data."
echo "========================================"
