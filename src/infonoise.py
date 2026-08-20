"""InfoNoise: online information-guided allocation of training noise levels.

Implements Algorithm 1 of "Noise Scheduling as Information-Guided Allocation in
Diffusion Training" (Raya et al., arXiv:2602.18647) for this repo's x-prediction
flow-matching setup.

The idea, in the paper's terms: with objective
    L(theta) = int pi(u) w(u) R_x(u; theta) du
the *effective allocation* is phi(u) = pi(u) w(u), and the principled target for
phi is the conditional-entropy-rate profile

    rho*(u) ∝ (1/2) |gamma'(u)| mmse(u),        gamma(u) = a(u)^2 / b(u)^2

(the Gaussian-channel I-MMSE identity, Eq. 10-11). mmse is unavailable during
training, so InfoNoise plugs in m_hat(u): a binned, smoothed estimate of the
*unweighted* x-space denoising loss, which every batch already computes. The
loss weight w stays fixed; only the sampling density pi moves:

    q_hat(u) ∝ (1/2)|gamma'(u)| m_hat(u)                            (Eq. 13)
    r(u)     = q_hat(u) g_{c,n}(u),   g_{c,n}(u) = u^n / (u^n + c^n) (Eq. 14)
    rho_hat  = r / int r
    pi(u)    ∝ rho_hat(u) / w(u)                                    (Eq. 15)

and u is drawn by inverse-CDF sampling off a tabulated grid (Eq. 16).

Coordinates. The paper works in the VE coordinate u = sigma, where
q_hat(sigma) ∝ m_hat(sigma)/sigma^3. This repo's forward process is the OT /
rectified-flow interpolation z = t*x0 + (1-t)*eps, so a(t)=t, b(t)=1-t and

    sigma(t) = (1 - t) / t,     t = 1 / (1 + sigma)

with t=1 clean and t=0 pure noise (note this is the *reverse* of the paper's t
convention in Eq. 44 -- only sigma is convention-free, which is why the whole
estimator below lives on a log-sigma grid and converts to t only at the end).
Checking the two agree: gamma(t) = t^2/(1-t)^2, |gamma'(t)| = 2t/(1-t)^3, so
q_hat(t) ∝ t*m_hat/(1-t)^3, which is exactly m_hat(sigma)/sigma^3 pushed through
|d sigma/d t| = 1/t^2. Densities here are per unit log-sigma (as in the paper's
figures), i.e. rho_hat_log(sigma) ∝ sigma * q_hat(sigma) = m_hat(sigma)/sigma^2.

The estimator is plain numpy on the host: it is a few hundred floats touched
once every `refresh_every` steps, so there is nothing to gain from putting it on
the accelerator, and keeping it off-device means the adapted density can change
shape without retriggering a JAX recompile.
"""

import json
import os
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from src.loss import T_CLIP


def sigma_of_t(t: np.ndarray) -> np.ndarray:
    """Noise-to-signal ratio b(t)/a(t) for z = t*x0 + (1-t)*eps."""
    return (1.0 - t) / np.maximum(t, 1e-12)


def t_of_sigma(sigma: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + sigma)


def loss_weight_of_sigma(sigma: np.ndarray) -> np.ndarray:
    """w(sigma): this repo's fixed x-prediction loss weight, in sigma coords.

    src/loss.py weights the per-pixel MSE by 1/max(T_CLIP, 1-t)^2, and
    1 - t = sigma / (1 + sigma).
    """
    one_minus_t = sigma / (1.0 + sigma)
    return 1.0 / np.maximum(T_CLIP, one_minus_t) ** 2


def _smooth_log_grid(values: np.ndarray, width_bins: float) -> np.ndarray:
    """Gaussian smoothing along the (uniformly log-spaced) bin axis."""
    if width_bins <= 0:
        return values
    radius = int(np.ceil(3.0 * width_bins))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / width_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


class InfoNoiseSampler:
    """Online estimator + inverse-CDF sampler for the training noise level.

    Usage per training step:
        t = sampler.sample_t(key, batch_size)      # (batch, 1, 1, 1)
        ... take the gradient step, get per-sample *unweighted* losses ...
        sampler.observe(t, per_sample_unweighted)
        sampler.maybe_refresh(global_step)

    All keyword arguments are exposed through ``train.py --dist_params`` so they
    land in the experiment name.

    Only two defaults here are actually pinned by the paper: the gate exponent
    ``gate_n = 3`` and the onset threshold ``gate_p = 0.002`` (Appendix B.6).
    The paper states that the grid, smoothing rule, minimum bin counts, warm-up
    length, and refresh cadence are held fixed within each domain, but never
    gives their values -- so ``num_bins``, ``sigma_min/max``, ``warmup_steps``,
    ``refresh_every``, ``ema_decay``, ``min_bin_count`` and ``smooth_bins``
    below are chosen for this repo (MNIST-scale runs, batch 128), not quoted.
    """

    def __init__(
        self,
        warmup_sample_fn: Callable[[PRNGKeyArray, int], Float[Array, "b 1 1 1"]],
        num_bins: int = 256,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        warmup_steps: int = 2000,
        refresh_every: int = 1000,
        ema_decay: float = 0.9,
        min_bin_count: int = 8,
        smooth_bins: float = 2.0,
        gate_n: float = 3.0,
        gate_c: Optional[float] = None,
        gate_p: float = 0.002,
        log_path: Optional[str] = None,
    ):
        if sigma_min <= 0 or sigma_max <= sigma_min:
            raise ValueError("need 0 < sigma_min < sigma_max")

        self.warmup_sample_fn = warmup_sample_fn
        self.num_bins = int(num_bins)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.warmup_steps = int(warmup_steps)
        self.refresh_every = int(refresh_every)
        self.ema_decay = float(ema_decay)
        self.min_bin_count = int(min_bin_count)
        self.smooth_bins = float(smooth_bins)
        self.gate_n = float(gate_n)
        self.gate_c = None if gate_c is None else float(gate_c)
        self.gate_p = float(gate_p)
        self.log_path = log_path

        # fixed grid, uniform in log sigma (the paper bins losses in log sigma)
        self.log_edges = np.linspace(
            np.log(self.sigma_min), np.log(self.sigma_max), self.num_bins + 1
        )
        self.log_centers = 0.5 * (self.log_edges[:-1] + self.log_edges[1:])
        self.centers = np.exp(self.log_centers)
        self.d_log = float(self.log_edges[1] - self.log_edges[0])

        # w is fixed by the objective, so it can be precomputed once
        self.weights = loss_weight_of_sigma(self.centers)

        # m_hat(u) === 1 at init (Algorithm 1, line 1); replaced wholesale at the
        # first refresh since "1" is not in the units of an actual denoising loss
        self.m_hat = np.ones(self.num_bins)
        self.acc_sum = np.zeros(self.num_bins)
        self.acc_count = np.zeros(self.num_bins)

        self.active = False  # False => still drawing from the warm-up prior pi_0
        self.num_refreshes = 0
        self._warned_gate = False
        self.last_gate_c: Optional[float] = None
        # pending device-side (t, loss) pairs, concatenated only at refresh time
        # so the training loop never blocks on a host sync
        self._pending: list = []

        # inverse-CDF table: cumulative mass at bin edges, sampled by
        # interpolating in log sigma within the containing bin
        self.pi = np.full(self.num_bins, 1.0 / (self.num_bins * self.d_log))
        self.cdf_edges = np.linspace(0.0, 1.0, self.num_bins + 1)

    # ---------------------------------------------------------------- sampling

    def sample_t(
        self, key: PRNGKeyArray, batch_size: int
    ) -> Float[Array, "batch 1 1 1"]:
        """Draw training noise levels, as t (1 = clean, 0 = pure noise)."""
        if not self.active:
            return self.warmup_sample_fn(key, batch_size)

        xi = np.asarray(jax.random.uniform(key, (batch_size,)), dtype=np.float64)
        # Eq. 16: u = F_pi^{-1}(xi), tabulated and inverted by interpolation
        log_sigma = np.interp(xi, self.cdf_edges, self.log_edges)
        t = t_of_sigma(np.exp(log_sigma))
        return jnp.asarray(t.reshape(-1, 1, 1, 1), dtype=jnp.float32)

    # ---------------------------------------------------------------- estimator

    def observe(self, t: Float[Array, "batch 1 1 1"], losses: Float[Array, " batch"]):
        """Record this batch's (noise level, unweighted x-space loss) pairs.

        ``losses`` must be the *unweighted* per-sample loss ||x0 - x_hat||^2 --
        the weighted loss is what trains the model, but the profile estimate
        needs the raw denoising error (Algorithm 1, line 4).
        """
        self._pending.append((t, losses))

    def _drain_pending(self):
        if not self._pending:
            return
        t = np.asarray(
            jnp.concatenate([jnp.reshape(p[0], (-1,)) for p in self._pending])
        ).astype(np.float64)
        losses = np.asarray(
            jnp.concatenate([jnp.reshape(p[1], (-1,)) for p in self._pending])
        ).astype(np.float64)
        self._pending = []

        sigma = sigma_of_t(t)
        finite = np.isfinite(sigma) & np.isfinite(losses)
        sigma, losses = sigma[finite], losses[finite]
        # samples outside the grid (only possible under a warm-up prior with
        # unbounded support) are clamped onto the end bins rather than dropped
        log_sigma = np.clip(
            np.log(np.maximum(sigma, 1e-12)),
            self.log_edges[0],
            self.log_edges[-1] - 1e-12,
        )
        idx = np.clip(
            np.searchsorted(self.log_edges, log_sigma, side="right") - 1,
            0,
            self.num_bins - 1,
        )
        np.add.at(self.acc_sum, idx, losses)
        np.add.at(self.acc_count, idx, 1.0)

    def maybe_refresh(self, step: int) -> bool:
        """Rebuild the sampler if this is a refresh step. Returns whether it did."""
        if step < self.warmup_steps:
            return False
        if (step - self.warmup_steps) % self.refresh_every != 0:
            return False
        self.refresh(step)
        return True

    def refresh(self, step: int = -1):
        """Algorithm 1, lines 5-6: smooth m_hat, form rho_hat, rebuild pi."""
        self._drain_pending()

        observed = self.acc_count >= self.min_bin_count
        if not observed.any():
            return

        raw = np.where(observed, self.acc_sum / np.maximum(self.acc_count, 1.0), 0.0)

        if self.num_refreshes == 0:
            # first estimate: take the measurement outright (m_hat === 1 is not a
            # meaningful prior in loss units) and fill unobserved bins by
            # interpolation in log sigma
            self.m_hat = np.interp(
                self.log_centers, self.log_centers[observed], raw[observed]
            )
        else:
            # smoothing across refreshes; bins with too few samples this window
            # keep their previous value rather than being pulled toward zero
            self.m_hat = np.where(
                observed,
                self.ema_decay * self.m_hat + (1.0 - self.ema_decay) * raw,
                self.m_hat,
            )

        self.m_hat = np.maximum(_smooth_log_grid(self.m_hat, self.smooth_bins), 0.0)
        self.acc_sum[:] = 0.0
        self.acc_count[:] = 0.0

        # Eq. 13. q_hat_sigma is the density per d sigma, q_hat the density per
        # d log sigma (= sigma * q_hat_sigma), which is the coordinate the
        # sampler and the paper's figures use.
        q_hat_sigma = self.m_hat / self.centers**3
        q_hat = self.m_hat / self.centers**2
        gate_c = self._resolve_gate_c(q_hat_sigma)
        gate = self.centers**self.gate_n / (
            self.centers**self.gate_n + gate_c**self.gate_n
        )
        r = q_hat * gate  # Eq. 14
        total = float(np.sum(r) * self.d_log)
        if not np.isfinite(total) or total <= 0:
            return
        rho_hat = r / total

        # Eq. 15: compensate the *fixed* loss weight so the induced allocation
        # phi = pi*w follows rho_hat. The objective is untouched.
        pi = rho_hat / self.weights
        pi = pi / (np.sum(pi) * self.d_log)

        self.pi = pi
        cdf = np.concatenate([[0.0], np.cumsum(pi) * self.d_log])
        self.cdf_edges = cdf / cdf[-1]
        self.last_gate_c = gate_c
        self.num_refreshes += 1
        self.active = True

        # The onset rule assumes a visible low-noise boundary segment in
        # m_hat/sigma^3. If it hasn't appeared yet (too short a warm-up, too few
        # bins), c lands near sigma_min, the gate does nothing, and pi collapses
        # onto the low-noise end -- the exact failure the gate exists to prevent.
        # Say so loudly rather than silently training on a degenerate schedule.
        if (
            self.gate_c is None
            and not self._warned_gate
            and gate_c <= self.centers[max(1, self.num_bins // 4)]
        ):
            self._warned_gate = True
            print(
                f"  WARNING: infonoise auto gate pivot c={gate_c:.4g} sits at the "
                f"low-noise end of the grid [{self.sigma_min:g}, {self.sigma_max:g}]; "
                "the profile estimate likely has no resolved boundary structure yet. "
                "Consider a longer warmup_steps or pinning gate_c (the paper pins "
                "c=0.2 for its non-image domains)."
            )

        self._log_refresh(step, rho_hat, q_hat)

    def _resolve_gate_c(self, r: np.ndarray) -> float:
        """Gate pivot c, by the onset-of-information rule (Appendix B.6).

        ``r`` is the *ungated* estimate per d sigma, r(sigma) = m_hat/sigma^3.
        The 1/sigma^3 path factor makes r blow up at the low-noise endpoint,
        where the estimate is least trustworthy (the denoiser is far from Bayes-
        optimal there early in training, and for near-discrete data the boundary
        shows up as a spurious power-law segment). The rule normalizes by the
        max, scans from high noise downward, and puts c at the first persistent
        crossing of p:

            r_bar = r / max r,   c = inf{ sigma : r_bar(sigma') < p for all sigma' >= sigma }

        i.e. the largest sigma still above threshold, which marks the upper edge
        of that boundary-dominated region. n = 3 and p = 0.002 are the paper's
        fixed values. Pass ``gate_c`` explicitly to pin it instead (the paper
        does this for its discrete/DNA runs, c = 0.2), or ``gate_c=0`` to
        disable gating entirely.
        """
        if self.gate_c is not None:
            return max(self.gate_c, 0.0)

        peak = float(np.max(r))
        if not np.isfinite(peak) or peak <= 0:
            return self.sigma_min
        # "for all sigma' >= sigma": scan down from the top so a single noisy bin
        # above threshold near the high-noise end can't drag c out with it
        above_from_top = (r / peak) >= self.gate_p
        idx = self.num_bins - 1
        while idx >= 0 and not above_from_top[idx]:
            idx -= 1
        if idx < 0:
            return self.sigma_min
        return float(self.centers[idx])

    # ---------------------------------------------------------------- reporting

    def profile_summary(self) -> dict:
        """Sigma quantiles of the current sampling density, for logging."""
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        sigmas = np.exp(np.interp(qs, self.cdf_edges, self.log_edges))
        ts = t_of_sigma(sigmas)
        return {
            "refreshes": self.num_refreshes,
            "gate_c": self.last_gate_c,  # None until the first refresh
            "pi_sigma_quantiles": {str(q): float(v) for q, v in zip(qs, sigmas)},
            "pi_t_quantiles": {str(q): float(v) for q, v in zip(qs, ts)},
        }

    def _log_refresh(self, step: int, rho_hat: np.ndarray, q_hat: np.ndarray):
        if self.log_path is None:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        record = {
            "step": int(step),
            "refresh": self.num_refreshes,
            "gate_c": self.last_gate_c,
            "sigma_centers": [float(v) for v in self.centers],
            "m_hat": [float(v) for v in self.m_hat],
            "q_hat_log_sigma": [float(v) for v in q_hat],
            "rho_hat": [float(v) for v in rho_hat],
            "pi": [float(v) for v in self.pi],
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
