"""Generate images from pure noise and show the sampler's state along the way,
for the gt_target model and the default model of one cell, from the same noise.

For each step of the Euler sampler (the same update as src.utils.sample_batch_x),
records the model input z_t and its prediction x0_hat. Rows per sample: z_t,
then x0_hat. Columns: a few steps spread over the trajectory, plus the final
sample.

    python scripts/plots/plot_sampling_trajectory.py --dataset mnist_x10 \
        --gt checkpoints/<gt run>_epoch_100.eqx --gt_t_clip 0.05 \
        --default checkpoints/<default run>_epoch_100.eqx --default_t_clip 0.05

Writes plots/sept30/sampling_<dataset>.png.
"""

import argparse
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
from src.schedules import get_shifted_steps  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--gt", required=True)
p.add_argument("--gt_t_clip", type=float, default=0.05)
p.add_argument("--default", required=True)
p.add_argument("--default_t_clip", type=float, default=0.05)
p.add_argument("--shift", type=float, default=0.3, help="0.3 = the shifted_coarse eval schedule")
p.add_argument("--num_steps", type=int, default=50)
p.add_argument("--n_samples", type=int, default=3)
p.add_argument("--n_cols", type=int, default=9)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_dir", default="plots/sept30")
args = p.parse_args()

spec = DATASETS[args.dataset]
shape = (spec.channels, spec.image_size, spec.image_size)
timesteps = get_shifted_steps(num_steps=args.num_steps, shift=args.shift)


def load(path):
    model = UNet(hidden_channels=256, num_channels=64, key=jax.random.PRNGKey(0), in_channels=1, num_classes=None)
    return eqx.tree_deserialise_leaves(path, model)


@eqx.filter_jit
def predict(model, z, t):
    return jax.vmap(model)(z, jnp.full((z.shape[0],), t))


def trajectory(model, t_clip):
    z = jax.random.normal(jax.random.PRNGKey(args.seed), (args.n_samples,) + shape)
    zs, x0s = [], []
    for i in range(args.num_steps):
        t, t_next = timesteps[i], timesteps[i + 1]
        x0 = predict(model, z, t)
        zs.append(np.asarray(z))
        x0s.append(np.asarray(x0))
        z = z + (t_next - t) * (x0 - z) / jnp.maximum(1 - t, t_clip)
    return np.stack(zs), np.stack(x0s), np.asarray(z)


runs = [
    ("gt target", *trajectory(load(args.gt), args.gt_t_clip)),
    ("default", *trajectory(load(args.default), args.default_t_clip)),
]

steps = np.linspace(0, args.num_steps - 1, args.n_cols).round().astype(int)
t_np = np.asarray(timesteps)
k = spec.sample_scale
n_rows = 2 * args.n_samples * len(runs)
fig, axes = plt.subplots(n_rows, args.n_cols + 1, figsize=(1.05 * (args.n_cols + 1), 1.1 * n_rows))
row = 0
for label, zs, x0s, final in runs:
    for s in range(args.n_samples):
        for j, step in enumerate(steps):
            axes[row, j].imshow(zs[step, s, 0], cmap="gray")
            axes[row + 1, j].imshow(np.clip(x0s[step, s, 0], -k, k), cmap="gray", vmin=-k, vmax=k)
            if row == 0:
                sigma = (1 - t_np[step]) / t_np[step] if t_np[step] > 0 else np.inf
                axes[row, j].set_title(f"step {step}\nσ={sigma:.3g}", fontsize=7)
        axes[row, -1].imshow(np.clip(final[s, 0], -k, k), cmap="gray", vmin=-k, vmax=k)
        axes[row + 1, -1].axis("off")
        if row == 0:
            axes[row, -1].set_title("final\nsample", fontsize=7)
        axes[row, 0].set_ylabel(f"{label}\nz_t", fontsize=7)
        axes[row + 1, 0].set_ylabel(f"{label}\nx̂₀", fontsize=7)
        row += 2
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(
    f"{args.dataset}: sampling from noise (shift {args.shift}, {args.num_steps} steps), same starting noise for both models",
    fontsize=9,
)
os.makedirs(args.out_dir, exist_ok=True)
out = os.path.join(args.out_dir, f"sampling_{args.dataset}.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
