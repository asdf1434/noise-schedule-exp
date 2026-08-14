"""Plot the training-time noise-level distributions (sample_t_* in src/schedules.py)
used in exp2, by drawing a large sample from each and histogramming.

Usage: python scripts/plots/plot_train_dists.py
"""

import jax
import matplotlib.pyplot as plt

from src.schedules import sample_t_logit_normal, sample_t_plateau_logit_normal, sample_t_uniform

N = 200_000
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 6)

dists = {
    "uniform": sample_t_uniform(keys[0], N),
    "logit_normal mu=0.0 sigma=1.0": sample_t_logit_normal(keys[1], N, mu=0.0, sigma=1.0),
    "logit_normal mu=0.0 sigma=0.3 (peaked)": sample_t_logit_normal(keys[2], N, mu=0.0, sigma=0.3),
    "logit_normal mu=1.5 sigma=1.0 (skew high/clean)": sample_t_logit_normal(keys[3], N, mu=1.5, sigma=1.0),
    "logit_normal mu=-1.5 sigma=1.0 (skew low/noisy)": sample_t_logit_normal(keys[4], N, mu=-1.5, sigma=1.0),
    "plateau mu=0.0 sigma=1.0 uniform_prob=0.3": sample_t_plateau_logit_normal(
        keys[5], N, mu=0.0, sigma=1.0, uniform_prob=0.3
    ),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
plt.style.use("seaborn-v0_8-whitegrid")

for ax, (name, samples) in zip(axes.flat, dists.items()):
    t = samples.reshape(-1)
    ax.hist(t, bins=100, range=(0, 1), density=True, color="steelblue", alpha=0.85)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("t  (0=noise, 1=clean)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)

fig.suptitle("Training-time noise level distributions (sample_t_*), N=200k samples each", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))

out_path = "plots/train_dists_comparison.png"
fig.savefig(out_path, dpi=200)
print(f"Saved to {out_path}")
