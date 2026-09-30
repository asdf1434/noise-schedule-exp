"""How a trained denoiser's output x0_hat changes with the noise level, against
the ground-truth target rho*(lambda).

For each of --n_images training images, one noise sample eps is fixed and the
noise level is swept over lambda = -2 log sigma. At each step the model sees
z = t*x + (1-t)*eps with t = 1/(1+sigma) and returns x0_hat. The speed
||d x0_hat / d lambda|| (per-pixel RMS, finite differences) should be largest
where rho* is largest.

    python scripts/plots/plot_denoising_trajectory.py \
        --checkpoint checkpoints/ds-mnist__cond-none__dist-gt_target_sigma_min_0.002_sigma_max_80.0__seed-0_epoch_100.eqx \
        --dataset mnist

Writes plots/sept30/trajectory_<dataset>_strip.png (x_t and x0_hat for a few images
at a few noise levels) and plots/sept30/trajectory_<dataset>_speed.png (rho* and the
speed on a shared lambda axis).
"""

import argparse
import json
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.datasets import DATASETS  # noqa: E402
from src.model import UNet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--sigma_min", type=float, default=0.002)
p.add_argument("--sigma_max", type=float, default=80.0)
p.add_argument("--n_images", type=int, default=16)
p.add_argument("--n_levels", type=int, default=121)
p.add_argument("--strip_images", type=int, default=4)
p.add_argument("--strip_levels", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_dir", default="plots/sept30")
args = p.parse_args()

spec = DATASETS[args.dataset]
model = UNet(hidden_channels=256, num_channels=64, key=jax.random.PRNGKey(0), in_channels=1, num_classes=None)
model = eqx.tree_deserialise_leaves(args.checkpoint, model)

rng = np.random.default_rng(args.seed)
images = np.asarray(spec.load(128))
x = jnp.asarray(images[rng.choice(len(images), args.n_images, replace=False)])
eps = jnp.asarray(rng.standard_normal(x.shape), dtype=x.dtype)

sigmas = np.exp(np.linspace(np.log(args.sigma_max), np.log(args.sigma_min), args.n_levels))
lam = -2 * np.log(sigmas)


@eqx.filter_jit
def denoise(model, z, t):
    return jax.vmap(model)(z, jnp.full((z.shape[0],), t))


x_t, x0_hat = [], []
for sigma in sigmas:
    t = 1.0 / (1.0 + sigma)
    z = t * x + (1 - t) * eps
    x_t.append(np.asarray(z))
    x0_hat.append(np.asarray(denoise(model, z, jnp.float32(t))))
x_t = np.stack(x_t)  # (levels, images, 1, h, w)
x0_hat = np.stack(x0_hat)

# per-image speed: RMS over pixels of the change between neighbouring levels
step = np.sqrt(np.mean(np.diff(x0_hat, axis=0) ** 2, axis=(2, 3, 4)))
speed = step / np.diff(lam)[:, None]
lam_mid = 0.5 * (lam[1:] + lam[:-1])

with open(f"results/q_target/{args.dataset}.json") as f:
    target = json.load(f)
os.makedirs(args.out_dir, exist_ok=True)

# rho* and speed, stacked on a shared lambda axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, constrained_layout=True)
ax1.plot(target["lambda"], target["q_lambda"], color="#2a6fdb", linewidth=2)
ax1.set_ylabel("ρ*(λ)")
ax1.set_title(f"{args.dataset}: ground-truth target and speed of x̂₀")
ax1.grid(alpha=0.3)
for s in speed.T:
    ax2.plot(lam_mid, s, color="#9aa7bd", linewidth=0.8)
ax2.plot(lam_mid, speed.mean(axis=1), color="#e8710a", linewidth=2, label=f"mean over {args.n_images} images")
ax2.set_ylabel("‖dx̂₀/dλ‖ (RMS per pixel)")
ax2.set_xlabel("λ = −2 log σ")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(lam.min(), lam.max())
fig.savefig(os.path.join(args.out_dir, f"trajectory_{args.dataset}_speed.png"), dpi=150)

# image strip: for each image, a row of x_t and a row of x0_hat
cols = np.linspace(0, args.n_levels - 1, args.strip_levels).round().astype(int)
k = spec.sample_scale
fig, axes = plt.subplots(
    2 * args.strip_images, args.strip_levels, figsize=(1.2 * args.strip_levels, 2.5 * args.strip_images)
)
for i in range(args.strip_images):
    for j, c in enumerate(cols):
        top, bottom = axes[2 * i, j], axes[2 * i + 1, j]
        top.imshow(x_t[c, i, 0], cmap="gray")
        bottom.imshow(np.clip(x0_hat[c, i, 0], -k, k), cmap="gray", vmin=-k, vmax=k)
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
        if i == 0:
            top.set_title(f"λ={lam[c]:.1f}", fontsize=8)
        if j == 0:
            top.set_ylabel("x_t", fontsize=8)
            bottom.set_ylabel("x̂₀", fontsize=8)
fig.suptitle(f"{args.dataset}: model input x_t and output x̂₀ as noise decreases (left to right)", fontsize=10)
fig.savefig(os.path.join(args.out_dir, f"trajectory_{args.dataset}_strip.png"), dpi=150, bbox_inches="tight")
mean_speed = np.convolve(speed.mean(axis=1), np.ones(5) / 5, mode="same")
target_peak = target["lambda"][int(np.argmax(target["q_lambda"]))]
print(f"rho* peak at lambda {target_peak:.2f}; mean speed (5-point smoothed) peaks at lambda {lam_mid[np.argmax(mean_speed)]:.2f}")
print(f"wrote {args.out_dir}/trajectory_{args.dataset}_{{speed,strip}}.png")
