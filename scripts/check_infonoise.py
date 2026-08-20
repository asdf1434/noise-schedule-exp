"""Self-check for src/infonoise.py. No GPU, no dataset, a few seconds.

Nothing here trains a model. The point is to verify the estimator in isolation,
against a channel whose answer is known analytically, so that when a real run
produces a weird-looking profile you know the bug is in the training setup and
not in the sampler.

The test channel is a unit-variance Gaussian source, for which

    mmse(sigma) = sigma^2 / (1 + sigma^2)

exactly. Feeding that (plus noise) in as the "denoising loss" means the target
allocation is known in closed form, so the learned pi can be compared to it
directly rather than merely eyeballed.

Usage:
    python scripts/check_infonoise.py
    python scripts/check_infonoise.py --verbose    # also print the sigma quantiles

Exits nonzero if any check fails.
"""

import argparse
import os
import sys

# so both `python scripts/check_infonoise.py` and `python -m scripts.check_infonoise`
# work from the repo root -- the former puts scripts/ on sys.path, not the root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from src.infonoise import InfoNoiseSampler, loss_weight_of_sigma, sigma_of_t
from src.schedules import sample_t_logit_normal, sample_t_uniform

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def true_mmse(sigma: np.ndarray) -> np.ndarray:
    return sigma**2 / (1.0 + sigma**2)


def run_synthetic(gate_c, steps=400, batch=256, noise_frac=0.05, seed=0):
    """Drive a sampler with losses drawn from the analytic mmse above."""
    sampler = InfoNoiseSampler(
        warmup_sample_fn=lambda k, b: sample_t_logit_normal(k, b, 0.0, 1.5),
        num_bins=192,
        warmup_steps=20,
        refresh_every=20,
        min_bin_count=4,
        gate_c=gate_c,
    )
    key = jax.random.PRNGKey(seed)
    for step in range(steps):
        key, k_t, k_noise = jax.random.split(key, 3)
        t = sampler.sample_t(k_t, batch)
        sigma = sigma_of_t(np.asarray(t).reshape(-1))
        jitter = 1.0 + noise_frac * np.asarray(jax.random.normal(k_noise, (batch,)))
        losses = true_mmse(sigma) * jitter
        sampler.observe(t, jnp.asarray(losses, dtype=jnp.float32))
        sampler.maybe_refresh(step)
    return sampler


def analytic_pi(sampler) -> np.ndarray:
    """The pi the sampler *should* converge to, given its own resolved gate c."""
    centers, c, n = sampler.centers, sampler.last_gate_c, sampler.gate_n
    gate = centers**n / (centers**n + c**n) if c else np.ones_like(centers)
    rho = true_mmse(centers) / centers**2 * gate
    rho /= rho.sum() * sampler.d_log
    pi = rho / loss_weight_of_sigma(centers)
    return pi / (pi.sum() * sampler.d_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n1. recovers the analytic allocation target")
    # gate_c=None exercises the paper's onset rule, 0.0 disables gating, 0.15 pins it.
    # Ungated is the loosest tolerance on purpose: with no gate the low-sigma bins
    # keep their 1/sigma^2 amplification and stay visibly noisier.
    for gate_c, tol in [(None, 0.10), (0.0, 0.25), (0.15, 0.10)]:
        s = run_synthetic(gate_c)
        err = float(np.abs(s.pi - analytic_pi(s)).max() / analytic_pi(s).max())
        label = "auto" if gate_c is None else f"c={gate_c}"
        detail = f"max rel err {err:.3f} (tol {tol}), c={s.last_gate_c:.4g}"
        if args.verbose:
            q = s.profile_summary()["pi_sigma_quantiles"]
            detail += f", sigma q25/50/75 = {q['0.25']:.3g}/{q['0.5']:.3g}/{q['0.75']:.3g}"
        check(f"gate {label}", err < tol, detail)

    print("\n2. draws actually follow pi")
    s = run_synthetic(0.15)
    key = jax.random.PRNGKey(1)
    t = np.asarray(s.sample_t(key, 200_000)).reshape(-1)
    hist, _ = np.histogram(np.log(sigma_of_t(t)), bins=s.log_edges, density=True)
    err = float(np.abs(hist - s.pi).max() / s.pi.max())
    check("empirical histogram matches pi", err < 0.06, f"max rel err {err:.3f}")
    check(
        "draws stay inside the grid",
        s.sigma_min <= sigma_of_t(t).min() and sigma_of_t(t).max() <= s.sigma_max,
        f"sigma in [{sigma_of_t(t).min():.4g}, {sigma_of_t(t).max():.4g}]",
    )

    print("\n3. densities normalize (per unit log sigma, not per bin)")
    check("sum(pi)*d_log == 1", abs(float(s.pi.sum() * s.d_log) - 1.0) < 1e-9)
    check("cdf ends at 1", abs(float(s.cdf_edges[-1]) - 1.0) < 1e-12)
    check("cdf is monotone", bool(np.all(np.diff(s.cdf_edges) >= 0)))

    print("\n4. warm-up reproduces the fixed prior exactly")
    # If this drifts, an infonoise run is no longer comparable to the fixed-dist
    # baseline it is supposed to start from.
    for name, fn in [("uniform", sample_t_uniform), ("logit_normal", sample_t_logit_normal)]:
        s = InfoNoiseSampler(warmup_sample_fn=fn, warmup_steps=10_000)
        k = jax.random.PRNGKey(7)
        check(
            f"warm-up == {name}",
            bool(jnp.array_equal(s.sample_t(k, 64), fn(k, 64))),
        )

    print("\n5. degenerate input leaves the previous sampler intact")
    s = run_synthetic(0.15)
    # drain first: run_synthetic stops mid-window, so without this the "NaN
    # window" would still carry the real batches observed since the last refresh
    s.refresh(step=-1)
    pi_before = s.pi.copy()
    s.observe(jnp.full((16, 1, 1, 1), 0.5), jnp.full((16,), jnp.nan))
    s.refresh(step=-1)
    check("all-NaN window doesn't clobber pi", bool(np.array_equal(s.pi, pi_before)))

    n = s.min_bin_count - 1  # too few samples in the one bin they land in
    s.observe(jnp.full((n, 1, 1, 1), 0.5), jnp.full((n,), 0.1))
    s.refresh(step=-1)
    check("under-occupied window doesn't clobber pi", bool(np.array_equal(s.pi, pi_before)))

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}\n")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
