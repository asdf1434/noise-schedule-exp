"""mmse(sigma) under the empirical prior, exactly as in Section B.4 of
arXiv:2602.18647 (InfoNoise).

    p_data(x0)   = (1/N) sum_i delta(x0 - x_i)                         (65)
    p(x; sigma)  = (1/N) sum_i N(x; x_i, sigma^2 I)                    (66)
    w_i(x)       = N(x; x_i, sigma^2 I) / sum_j N(x; x_j, sigma^2 I)   (67)
    x_hat(x)     = sum_i w_i(x) x_i                                    (68)
    Cov(x0 | x)  = sum_i w_i(x) (x_i - x_hat)(x_i - x_hat)^T           (69)
    mmse(sigma)  = E_{x ~ p(.; sigma)} [ tr Cov(x0 | x) ]              (70)

The expectation in (70) is the only thing not in closed form. It is estimated
by Monte Carlo: draw i uniformly from the N images, set x = x_i + sigma * eps.
The {x_i} are the full training split, as the repo loads it for training.

mmse is the squared norm over all D pixels, as in Eq. (60). Divide by D to
compare with the per-pixel training loss.

--class_label c restricts {x_i} to the images of class c, which gives the mmse
of a denoiser that knows the class.

Usage:
    python scripts/analysis/mmse_b4.py --dataset mnist
    python scripts/analysis/mmse_b4.py --dataset cifar10_x10
    python scripts/analysis/mmse_b4.py --dataset mnist --class_label 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_images(dataset: str, class_label=None) -> np.ndarray:
    """(N, D) float64, the training split in the repo's [-1, 1] format times the
    dataset's scale. With class_label, only that class's images."""
    from src.datasets import DATASETS

    if class_label is None:
        images = np.asarray(DATASETS[dataset].load(128))
    else:
        images, labels = DATASETS[dataset].load(128, with_labels=True)
        images = np.asarray(images)[np.asarray(labels) == class_label]
    return images.reshape(images.shape[0], -1).astype(np.float64)


def mmse_at_sigma(xs, sq_norms, sigma, rng, n_mc, block):
    """Monte-Carlo estimate of Eq. (70). Returns (mean, standard error)."""
    n = xs.shape[0]
    tr_cov = np.empty(n_mc)
    for start in range(0, n_mc, block):
        b = min(block, n_mc - start)
        x = xs[rng.integers(0, n, size=b)] + sigma * rng.standard_normal((b, xs.shape[1]))

        # log N(x; x_i, sigma^2 I) up to a constant shared by all i, Eq. (67).
        # float64 is needed: ||x - x_i||^2 is expanded as ||x||^2 - 2 x.x_i + ||x_i||^2,
        # and the difference is divided by 2 sigma^2.
        sq_dist = sq_norms[None, :] - 2.0 * (x @ xs.T) + np.einsum("ij,ij->i", x, x)[:, None]
        log_w = -sq_dist / (2.0 * sigma**2)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)

        x_hat = w @ xs  # Eq. (68)
        # tr Cov, Eq. (69): sum_i w_i ||x_i||^2 - ||x_hat||^2. Rounding can leave
        # a value of order 1e-12 below zero when w is one-hot; those are set to 0.
        tr = w @ sq_norms - np.einsum("ij,ij->i", x_hat, x_hat)
        tr_cov[start : start + b] = np.maximum(tr, 0.0)

    return float(tr_cov.mean()), float(tr_cov.std(ddof=1) / np.sqrt(n_mc))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="name in src.datasets.DATASETS")
    p.add_argument("--class_label", type=int, default=None)
    p.add_argument("--sigma_min", type=float, default=1e-3)
    p.add_argument("--sigma_max", type=float, default=1e3)
    p.add_argument("--per_decade", type=int, default=10)
    p.add_argument("--n_mc", type=int, default=8192, help="Monte-Carlo draws per sigma")
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results/mmse_b4")
    args = p.parse_args()

    xs = load_images(args.dataset, args.class_label)
    name = args.dataset if args.class_label is None else f"{args.dataset}_class{args.class_label}"
    sq_norms = np.einsum("ij,ij->i", xs, xs)
    n_decades = np.log10(args.sigma_max / args.sigma_min)
    sigmas = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), int(round(n_decades * args.per_decade)) + 1
    )
    print(f"{name}: N={xs.shape[0]} D={xs.shape[1]}, {len(sigmas)} sigmas, n_mc={args.n_mc}")

    mmse, stderr = [], []
    t0 = time.time()
    for k, sigma in enumerate(sigmas):
        rng = np.random.default_rng([args.seed, k])
        m, se = mmse_at_sigma(xs, sq_norms, sigma, rng, args.n_mc, args.block)
        mmse.append(m)
        stderr.append(se)
        print(f"  sigma={sigma:<9.4g} mmse={m:<12.5g} se={se:<10.3g} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "class_label": args.class_label,
                "n": int(xs.shape[0]),
                "dim": int(xs.shape[1]),
                "n_mc": args.n_mc,
                "seed": args.seed,
                "sigma": sigmas.tolist(),
                "mmse": mmse,
                "mmse_stderr": stderr,
            },
            f,
            indent=1,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
