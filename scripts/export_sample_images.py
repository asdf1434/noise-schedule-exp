# WRITTEN BY CLAUDE

"""Copies a random subset of generated sample images out of eval_runs/ into a
small, git-addable folder -- so you can pull just a handful of images to your
own machine to eyeball, instead of the whole (huge, gitignored) eval_runs/
tree.

For one dataset, picks one seed per training distribution (the lowest seed
that has data), then for every (train_dist, sampling_schedule) cell, samples
N_PER_CELL epochs (evenly spaced across whatever epoch_* dirs exist) x
N_PER_EPOCH images each, and copies them to:

    eval_runs/export_imgs/<dataset>/<dist>/epoch_<E>/<schedule>/<seed>_<orig_filename>

Default counts (6 dists x 4 schedules x 10 epochs x 5 images) = 1200 images.

Usage:
    python scripts/export_sample_images.py --dataset eurosat64
    python scripts/export_sample_images.py --dataset eurosat64 --n_epochs 10 --n_per_epoch 5
"""

import argparse
import glob
import os
import random
import re
import shutil

from src.naming import parse_exp_name

SCHEDULES = ["uniform", "shifted_fine", "shifted_coarse", "logit_normal"]

DISTS = [
    "uniform",
    "logit_normal_mu_0.0_sigma_1.0",
    "logit_normal_mu_0.0_sigma_0.3",
    "logit_normal_mu_1.5_sigma_1.0",
    "logit_normal_mu_-1.5_sigma_1.0",
    "plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]


def pick_seed_dir(dataset: str, dist: str) -> str | None:
    """Among eval_runs/ds-<dataset>__cond-*__dist-<dist>__seed-*, return the
    one with the lowest seed number that actually has epoch_* subdirs."""
    candidates = []
    for path in glob.glob(os.path.join("eval_runs", f"ds-{dataset}__cond-*__dist-{dist}__seed-*")):
        name = os.path.basename(path)
        try:
            seed = parse_exp_name(name)["seed"]
        except ValueError:
            continue
        if glob.glob(os.path.join(path, "epoch_*")):
            candidates.append((seed, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def pick_epochs(exp_dir: str, n_epochs: int) -> list[str]:
    """Evenly-spaced subset of this experiment's epoch_* dirs (by epoch number)."""
    epoch_dirs = glob.glob(os.path.join(exp_dir, "epoch_*"))
    numbered = []
    for d in epoch_dirs:
        m = re.match(r"epoch_(\d+)$", os.path.basename(d))
        if m:
            numbered.append((int(m.group(1)), d))
    numbered.sort()
    if not numbered:
        return []
    if len(numbered) <= n_epochs:
        return [d for _, d in numbered]
    # evenly spaced indices across the sorted list, including both ends
    idxs = [round(i * (len(numbered) - 1) / (n_epochs - 1)) for i in range(n_epochs)]
    idxs = sorted(set(idxs))
    return [numbered[i][1] for i in idxs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eurosat64", help="Dataset name as it appears in exp_name (ds-<dataset>__...)")
    parser.add_argument("--dists", nargs="+", default=DISTS, help="Train-dist tokens to sample (dist-<token> in exp_name)")
    parser.add_argument("--schedules", nargs="+", default=SCHEDULES, help="Sampling schedule subfolder names")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epoch checkpoints to sample per (dist, schedule)")
    parser.add_argument("--n_per_epoch", type=int, default=5, help="Number of random images to sample per (dist, schedule, epoch)")
    parser.add_argument("--dest", default=None, help="Output dir (default: eval_runs/export_imgs/<dataset>)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image sampling (reproducible picks)")
    args = parser.parse_args()

    dest_root = args.dest or os.path.join("eval_runs", "export_imgs", args.dataset)
    rng = random.Random(args.seed)

    total_copied = 0
    for dist in args.dists:
        exp_dir = pick_seed_dir(args.dataset, dist)
        if exp_dir is None:
            print(f"[SKIP] no eval_runs dir found for dataset={args.dataset} dist={dist}")
            continue
        seed_used = parse_exp_name(os.path.basename(exp_dir))["seed"]
        epoch_dirs = pick_epochs(exp_dir, args.n_epochs)
        if not epoch_dirs:
            print(f"[SKIP] {exp_dir} has no epoch_* dirs")
            continue

        print(f"{dist}: using seed {seed_used}, {len(epoch_dirs)} epochs -> {[os.path.basename(d) for d in epoch_dirs]}")

        for epoch_dir in epoch_dirs:
            epoch_name = os.path.basename(epoch_dir)
            for schedule in args.schedules:
                sched_dir = os.path.join(epoch_dir, schedule)
                images = sorted(glob.glob(os.path.join(sched_dir, "*.png")))
                if not images:
                    print(f"  [SKIP] no images in {sched_dir}")
                    continue
                chosen = rng.sample(images, min(args.n_per_epoch, len(images)))

                out_dir = os.path.join(dest_root, dist, epoch_name, schedule)
                os.makedirs(out_dir, exist_ok=True)
                for src_path in chosen:
                    out_name = f"seed{seed_used}_{os.path.basename(src_path)}"
                    shutil.copy2(src_path, os.path.join(out_dir, out_name))
                total_copied += len(chosen)

    print(f"\nCopied {total_copied} images -> {dest_root}")
    print(f"git add {dest_root}")


if __name__ == "__main__":
    main()
