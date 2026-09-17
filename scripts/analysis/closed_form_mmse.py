"""Closed-form MMSE(sigma) for a dataset, the ground-truth target InfoNoise estimates.

InfoNoise's target profile is rho*(sigma) ∝ (1/2)|gamma'(sigma)| mmse(sigma), with
the unavailable mmse replaced during training by m_hat: the model's own binned,
smoothed, unweighted denoising loss. On a dataset small enough to hold in memory
the true mmse is available in closed form, which lets the two failure modes be
told apart -- a bad estimate of a good target, or a good estimate of a bad one.

The closed form. Treat the dataset as the empirical distribution: x is uniform
over the N training images. For the Gaussian channel y = x + sigma*eps the
posterior is then a softmax over those images,

    E[x | y] = sum_i w_i x_i,    w_i  ∝  exp( -||y - x_i||^2 / (2 sigma^2) )

which is exact, not an approximation -- it is Bayes' rule on a finite mixture.
mmse(sigma) = E ||x - E[x|y]||^2 / D follows by Monte Carlo over (x, eps). The
division by D matches src.loss.compute_loss_cond, whose per-sample unweighted
statistic (the thing InfoNoise bins into m_hat) is a mean over pixels, so the
two curves are directly comparable and share the ceiling mmse(sigma) <= sigma^2.

Two variants are computed, because the empirical distribution is not the data
distribution:

  "empirical"  x drawn from the same images the posterior is built from. Exactly
               the MMSE of the empirical distribution. Once sigma falls below
               the typical distance between neighbouring images the softmax
               collapses onto the single nearest one and this falls to zero far
               faster than the real quantity does, so its low-sigma tail is an
               artifact of the finite dataset.
  "holdout"    posterior built from the support images, x drawn from disjoint
               held-out ones. Does not collapse to zero -- instead it saturates
               at the mean squared distance to the nearest training image, which
               for MNIST is ~0.096 per pixel and sits far ABOVE the sigma^2
               ceiling at small sigma. An upper bound on the true mmse, and a
               very loose one down there.
  "gaussian"   mmse of the Gaussian with the data's own mean and covariance,
               (1/D) sum_i lambda_i sigma^2 / (lambda_i + sigma^2) over the
               covariance eigenvalues. Exact for that distribution, so unlike
               the other two it obeys mmse <= sigma^2 everywhere and stays
               smooth at small sigma. It is an upper bound on the true mmse
               (the Gaussian is the maximum-entropy distribution at fixed
               covariance) and is the only one of the three that is informative
               in the low-noise regime the gate depends on.

Why three. In high dimension the empirical MMSE is close to a step function: as
long as sigma^2 * D is well below the squared distance between distinct images,
the posterior puts essentially all its mass on the true image and the error is
exactly zero. On MNIST that holds out to sigma ~ 0.6, which covers most of the
range the schedules care about. So the empirical and held-out curves bracket the
truth only at moderate and high noise; below that the only statements available
are the sigma^2 ceiling and the Gaussian reference.

Scaled datasets (mnist_x10, cifar10_x0.1, ...) need no separate run. With
x' = k*x the posterior weights are unchanged under sigma' = k*sigma, so

    mmse_k(sigma) = k^2 * mmse_1(sigma / k)

exactly; --verify-scaling checks this numerically. Only the base datasets are
computed here, and consumers rescale.

Usage:
    python scripts/analysis/closed_form_mmse.py --dataset mnist
    python scripts/analysis/closed_form_mmse.py --dataset cifar10 --n_mc 1024
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# runnable as `python scripts/analysis/closed_form_mmse.py`, like the rest of
# scripts/, which means the repo root is not on the path by default
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _load_flat(dataset: str) -> np.ndarray:
    """(N, D) float64 view of a dataset in the repo's [-1, 1] NCHW format."""
    from src.datasets import DATASETS

    images = np.asarray(DATASETS[dataset].load(128))
    return images.reshape(images.shape[0], -1).astype(np.float64)


def mmse_at_sigma(
    support: np.ndarray,
    targets: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    n_mc: int,
    block: int = 256,
) -> float:
    """Monte-Carlo E||x - E[x|y]||^2 / D with x drawn from ``targets`` and the
    posterior taken over ``support``.

    float64 throughout on purpose. The Gram expansion
    ||y - x_i||^2 = ||y||^2 - 2 y.x_i + ||x_i||^2 cancels two terms of order D
    against a residual that is of order sigma^2 * D, and that residual is then
    divided by 2*sigma^2. In float32 the cancellation error alone reaches
    several units of log-weight at sigma = 0.002, which silently destroys the
    softmax at exactly the noise levels this script exists to measure.
    """
    n_support, dim = support.shape
    sq_support = np.einsum("ij,ij->i", support, support)
    support_t = support.T

    idx = rng.integers(0, targets.shape[0], size=n_mc)
    total_sq_err = 0.0
    for start in range(0, n_mc, block):
        x = targets[idx[start : start + block]]
        y = x + sigma * rng.standard_normal(x.shape)

        # (B, N) squared distances, then a row-stabilized softmax
        d2 = sq_support[None, :] - 2.0 * (y @ support_t)
        d2 += np.einsum("ij,ij->i", y, y)[:, None]
        logw = d2 * (-0.5 / sigma**2)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)

        x_hat = w @ support
        total_sq_err += float(np.einsum("ij,ij->", x - x_hat, x - x_hat))

    return total_sq_err / (n_mc * dim)


def gaussian_mmse(flat: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """mmse(sigma) per pixel for N(mean, cov) fitted to ``flat``.

    For a Gaussian prior the posterior mean is linear and the error covariance
    is (Sigma^-1 + I/sigma^2)^-1, whose trace diagonalizes over the eigenvalues
    of Sigma as sum_i lambda_i sigma^2 / (lambda_i + sigma^2).
    """
    centered = flat - flat.mean(axis=0, keepdims=True)
    n, dim = centered.shape
    # eigenvalues of the covariance via the Gram matrix when N < D, else directly
    if n < dim:
        gram = centered @ centered.T / (n - 1)
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.concatenate([eigenvalues, np.zeros(dim - n)])
    else:
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / (n - 1))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    s2 = np.asarray(sigmas)[:, None] ** 2
    return (eigenvalues[None, :] * s2 / (eigenvalues[None, :] + s2)).sum(axis=1) / dim


def compute_curve(dataset: str, args) -> dict:
    rng = np.random.default_rng(args.seed)
    flat = _load_flat(dataset)
    perm = rng.permutation(flat.shape[0])
    holdout = flat[perm[: args.holdout]]
    support = flat[perm[args.holdout :]]

    sigmas = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.num_sigma
    )
    out = {"empirical": [], "holdout": []}
    t0 = time.time()
    for i, sigma in enumerate(sigmas):
        out["empirical"].append(
            mmse_at_sigma(support, support, sigma, rng, args.n_mc, args.block)
        )
        out["holdout"].append(
            mmse_at_sigma(support, holdout, sigma, rng, args.n_mc, args.block)
        )
        print(
            f"  [{i + 1:>3}/{len(sigmas)}] sigma={sigma:<10.5g} "
            f"empirical={out['empirical'][-1]:<12.6g} "
            f"holdout={out['holdout'][-1]:<12.6g} "
            f"ceiling={sigma**2:<12.6g} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    return {
        "dataset": dataset,
        "mmse_gaussian": gaussian_mmse(flat, sigmas).tolist(),
        "dim": int(flat.shape[1]),
        "n_support": int(support.shape[0]),
        "n_holdout": int(holdout.shape[0]),
        "n_mc": args.n_mc,
        "seed": args.seed,
        "sigma": sigmas.tolist(),
        "mmse_empirical": out["empirical"],
        "mmse_holdout": out["holdout"],
    }


def verify_scaling(dataset: str, args) -> None:
    """Check mmse_k(sigma) = k^2 mmse_1(sigma/k) numerically, so downstream code
    can rescale a single base curve instead of recomputing per k."""
    rng = np.random.default_rng(args.seed)
    flat = _load_flat(dataset)[: args.verify_subset]
    print(f"scaling check on {dataset} ({flat.shape[0]} images)")
    for k in (0.1, 10.0):
        for sigma in (0.05, 1.0, 20.0):
            base = mmse_at_sigma(
                flat, flat, sigma, np.random.default_rng(args.seed), 256, args.block
            )
            scaled = mmse_at_sigma(
                flat * k,
                flat * k,
                sigma * k,
                np.random.default_rng(args.seed),
                256,
                args.block,
            )
            rel = abs(scaled - k**2 * base) / max(k**2 * base, 1e-300)
            status = "OK " if rel < 1e-6 else "BAD"
            print(
                f"  [{status}] k={k:<5g} sigma={sigma:<6g} "
                f"k^2*mmse_1(sigma/k)={k**2 * base:.10g}  "
                f"mmse_k(sigma)={scaled:.10g}  rel err {rel:.2e}"
            )
    _ = rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mnist", help="base dataset (unscaled)")
    p.add_argument("--sigma_min", type=float, default=0.001)
    p.add_argument("--sigma_max", type=float, default=160.0)
    p.add_argument("--num_sigma", type=int, default=72)
    p.add_argument("--n_mc", type=int, default=1024, help="Monte-Carlo draws per sigma")
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--block", type=int, default=256, help="targets per matmul block")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/closed_form_mmse")
    p.add_argument("--verify-scaling", action="store_true")
    p.add_argument("--verify_subset", type=int, default=4000)
    args = p.parse_args()

    if args.verify_scaling:
        verify_scaling(args.dataset, args)
        return

    print(f"closed-form MMSE for {args.dataset}")
    record = compute_curve(args.dataset, args)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
