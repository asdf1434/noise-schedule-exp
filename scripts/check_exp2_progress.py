"""Quick progress check for run_exp2.sh's 80-task sweep (4 train dists x 20 seeds).

Reads checkpoints/ to find the latest completed epoch per (dist, seed), since
train.py only writes a checkpoint at each --eval_interval (every 10 epochs by
default) and the final epoch. Also greps slurm logs for errors.

Usage: python check_exp2_progress.py
"""

import glob
import os
import re

EPOCHS_TOTAL = 100

# base_name -> exp_name prefix, matching train.py's auto-naming
# (base_name = f"{train_dist}_{param_str}", param_str built from dist_params
# dict in insertion order)
DIST_BASE_NAMES = {
    "logit_normal_peaked": "logit_normal_mu_0.0_sigma_0.3",
    "logit_normal_skew_hi": "logit_normal_mu_1.5_sigma_1.0",
    "logit_normal_skew_lo": "logit_normal_mu_-1.5_sigma_1.0",
    "plateau": "plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
}
NUM_SEEDS = 20

CKPT_EPOCH_RE = re.compile(r"_epoch_(\d+)\.eqx$")


def latest_epoch(base_name: str, seed: int, checkpoint_dir: str = "checkpoints") -> int:
    exp_name = f"{base_name}_seed{seed}"
    pattern = os.path.join(checkpoint_dir, f"{exp_name}_epoch_*.eqx")
    epochs = []
    for path in glob.glob(pattern):
        m = CKPT_EPOCH_RE.search(path)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else 0


def scan_errors(log_glob: str = "logs/slurm/slurm_exp2_train_dists_*.out") -> list[str]:
    bad = []
    for path in glob.glob(log_glob):
        try:
            with open(path, errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        if "Error" in text or "Traceback" in text:
            bad.append(path)
    return sorted(bad)


def main():
    total_done = 0
    total_started = 0
    total_expected = len(DIST_BASE_NAMES) * NUM_SEEDS

    for label, base_name in DIST_BASE_NAMES.items():
        print(f"\n{label}  ({base_name})")
        row = []
        for seed in range(NUM_SEEDS):
            epoch = latest_epoch(base_name, seed)
            if epoch >= EPOCHS_TOTAL:
                total_done += 1
            if epoch > 0:
                total_started += 1
            row.append(f"{epoch:>3}" if epoch else "  -")
        print("  seeds 0-19: " + " ".join(row))

    print(
        f"\n{total_done}/{total_expected} runs fully complete "
        f"(epoch {EPOCHS_TOTAL}), {total_started}/{total_expected} started"
    )

    errors = scan_errors()
    if errors:
        print(f"\n{len(errors)} slurm log(s) contain 'Error'/'Traceback':")
        for path in errors:
            print(f"  {path}")
    else:
        print("\nNo errors found in slurm logs (logs/slurm/slurm_exp2_train_dists_*.out).")


if __name__ == "__main__":
    main()
