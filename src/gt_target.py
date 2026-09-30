"""Fixed training distribution pi = rho*/w built from the exact B.4 mmse.

rho* is the paper's target profile (arXiv:2602.18647, Eq. 61) with the Bayes
mmse of the empirical prior (Appendix B.4, results/mmse_b4/), and w is this
run's fixed loss weight. Per unit log sigma,

    pi(sigma) ∝ mmse(sigma) / sigma^2 / w(sigma)

restricted to [sigma_min, sigma_max] and drawn by inverse CDF. mmse is
interpolated linearly in log-log between the 61 grid points of the B.4 file.

With per_class=True, each training example draws its noise level from its own
class's pi, built from results/mmse_b4/<dataset>_class<c>.json.
"""

import json
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

from src.infonoise import loss_weight_of_sigma, t_of_sigma

NUM_POINTS = 4096


def _log_sigma_cdf(path, log_sigma, loss_weighting, t_clip, sigma_data):
    """CDF of pi on the log-sigma points, from one B.4 mmse file."""
    with open(path) as f:
        b4 = json.load(f)
    log_mmse = np.log(np.maximum(np.array(b4["mmse"]), 1e-300))
    mmse = np.exp(np.interp(log_sigma, np.log(b4["sigma"]), log_mmse))
    sigma = np.exp(log_sigma)
    pi = mmse / sigma**2 / loss_weight_of_sigma(sigma, loss_weighting, t_clip, sigma_data)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pi[1:] + pi[:-1]) * np.diff(log_sigma))])
    if cdf[-1] <= 0:
        raise ValueError(f"{path}: mmse is zero over the whole sigma range")
    return cdf / cdf[-1]


class GtTargetSampler:
    def __init__(
        self,
        dataset: str,
        loss_weighting: str,
        t_clip: float,
        sigma_data: float,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        per_class: bool = False,
        num_classes: int = 10,
        mmse_dir: str = "results/mmse_b4",
    ):
        self.per_class = bool(per_class)
        self.log_sigma = np.linspace(np.log(sigma_min), np.log(sigma_max), NUM_POINTS)
        names = [f"{dataset}_class{c}" for c in range(num_classes)] if self.per_class else [dataset]
        self.cdfs = np.stack(
            [
                _log_sigma_cdf(
                    os.path.join(mmse_dir, f"{name}.json"),
                    self.log_sigma,
                    loss_weighting,
                    t_clip,
                    sigma_data,
                )
                for name in names
            ]
        )

    def sample_t(
        self, key: PRNGKeyArray, batch_size: int, labels: Optional[Int[Array, " batch"]] = None
    ) -> Float[Array, "batch 1 1 1"]:
        """Draw training noise levels, as t (1 = clean, 0 = pure noise)."""
        xi = np.asarray(jax.random.uniform(key, (batch_size,)), dtype=np.float64)
        rows = np.zeros(batch_size, dtype=int) if not self.per_class else np.asarray(labels)
        log_sigma = np.empty(batch_size)
        for row in np.unique(rows):
            mask = rows == row
            log_sigma[mask] = np.interp(xi[mask], self.cdfs[row], self.log_sigma)
        t = t_of_sigma(np.exp(log_sigma))
        return jnp.asarray(t.reshape(-1, 1, 1, 1), dtype=jnp.float32)
