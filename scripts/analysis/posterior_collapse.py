"""Evidence that the Appendix B.4 empirical-Bayes MMSE is zero by construction.

B.4 treats the dataset as N point masses, so the Bayes denoiser is a softmax over
the training images. In high dimension that softmax is decided long before the
noise is visible: the log-odds favouring the true image go as d^2/(2 sigma^2),
with d the distance to the nearest other image, and that has to fall below log N
before the posterior spreads at all.

This measures the three quantities that show it, so the claim rests on numbers
rather than on the argument:

  max posterior weight      1.000000 means the posterior IS a point mass
  effective # images        1/sum(w^2), the number being averaged over
  fraction of ambiguous     draws where max weight < 0.99; zero means every term
    draws                   in the Eq. 70 average is exactly zero, so no amount
                            of averaging produces a nonzero answer

Usage:
    python scripts/analysis/posterior_collapse.py --dataset mnist
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def measure(support, sigma, rng, n_draws, block=128):
    sq = np.einsum("ij,ij->i", support, support)
    dim = support.shape[1]
    idx = rng.integers(0, support.shape[0], n_draws)
    max_w, eff_n, tr_cov = [], [], []
    for start in range(0, n_draws, block):
        x = support[idx[start : start + block]]
        y = x + sigma * rng.standard_normal(x.shape)
        d2 = sq[None, :] - 2.0 * (y @ support.T)
        d2 += np.einsum("ij,ij->i", y, y)[:, None]
        logw = d2 * (-0.5 / sigma**2)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        x_hat = w @ support
        max_w.append(w.max(axis=1))
        eff_n.append(1.0 / np.einsum("ij,ij->i", w, w))
        # Eq. 69 trace, as sum_i w_i||x_i||^2 - ||x_hat||^2
        tr_cov.append((w @ sq) - np.einsum("ij,ij->i", x_hat, x_hat))
    max_w = np.concatenate(max_w)
    return {
        "sigma": float(sigma),
        "max_weight": float(max_w.mean()),
        "effective_n": float(np.concatenate(eff_n).mean()),
        "mmse": float(np.concatenate(tr_cov).mean() / dim),
        "frac_ambiguous": float((max_w < 0.99).mean()),
        "n_draws": int(n_draws),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--n_draws", type=int, default=512)
    p.add_argument("--num_sigma", type=int, default=40)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/posterior_collapse")
    args = p.parse_args()

    from src.datasets import DATASETS

    images = np.asarray(DATASETS[args.dataset].load(128))
    support = images.reshape(images.shape[0], -1).astype(np.float64)

    rng = np.random.default_rng(args.seed)
    # distance to the nearest OTHER image, on a subsample -- the scale that sets
    # where the collapse ends
    sub = support[rng.choice(support.shape[0], 400, replace=False)]
    sq = np.einsum("ij,ij->i", support, support)
    d2 = sq[None, :] - 2.0 * (sub @ support.T)
    d2 += np.einsum("ij,ij->i", sub, sub)[:, None]
    np.put_along_axis(d2, d2.argmin(axis=1)[:, None], np.inf, axis=1)
    nn = float(np.sqrt(d2.min(axis=1)).mean())

    sigmas = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.num_sigma
    )
    rows = []
    for sigma in sigmas:
        row = measure(support, sigma, rng, args.n_draws)
        rows.append(row)
        print(
            f"  sigma={sigma:<9.4g} max_w={row['max_weight']:<10.6f} "
            f"eff_n={row['effective_n']:<9.1f} ambiguous={row['frac_ambiguous']:<7.3f} "
            f"mmse={row['mmse']:.4g}",
            flush=True,
        )

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "dim": int(support.shape[1]),
                "n_images": int(support.shape[0]),
                "nearest_neighbour_distance": nn,
                "rows": rows,
            },
            f,
        )
    print(f"wrote {path}   (mean nearest-neighbour distance {nn:.2f})")


if __name__ == "__main__":
    main()
