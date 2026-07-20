import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from PIL import Image

from src.conditioning import build_cond_channels, inject_known

# Dataset loading lives in src/datasets.py now (registry + per-dataset loaders);
# re-exported here so existing `from src.utils import get_dataloaders` callers
# keep working.
from src.datasets import get_dataloaders  # noqa: F401


def save_images(samples: np.ndarray, folder_name: str):
    os.makedirs(folder_name, exist_ok=True)
    # convert from [-1, 1] to [0, 255] for all images
    samples = ((samples + 1) * 127.5).clip(0, 255).astype(np.uint8)

    samples = samples.squeeze(1)
    samples_rgb = np.stack([samples, samples, samples], axis=-1)
    # save all in a single batch
    for i, img in enumerate(samples_rgb):
        Image.fromarray(img).save(os.path.join(folder_name, f"{i:05d}.png"))


@eqx.filter_jit
def sample_batch_x(
    model: eqx.Module,
    key: PRNGKeyArray,
    timesteps: Float[Array, " steps_plus_1"],
    batch_size: int,
    image_shape: tuple = (1, 28, 28),
) -> Float[Array, "batch c h w"]:
    """
    sample from model by predicting x
    input is the timestep sequence

    uses lax.scan (instead of a python for loop) so the whole sampling
    trajectory compiles into one fused call rather than dispatching each
    step eagerly
    """
    # Initialize with pure noise
    z = jax.random.normal(key, (batch_size,) + image_shape)

    def step(
        z: Float[Array, "batch c h w"], ts: Float[Array, " 2"]
    ) -> tuple[Float[Array, "batch c h w"], None]:
        t, t_next = ts[0], ts[1]
        t_batched = jnp.full(batch_size, t)

        # predict clean data
        x_pred = jax.vmap(model)(z, t_batched)

        # calculate velocity
        v = (x_pred - z) / jnp.maximum(1 - t, 0.05)

        z = z + (t_next - t) * v
        return z, None

    ts_pairs = jnp.stack([timesteps[:-1], timesteps[1:]], axis=1)
    z, _ = jax.lax.scan(step, z, ts_pairs)

    return z


def sample_batch_cond(
    model: eqx.Module,
    key: PRNGKeyArray,
    timesteps: Float[Array, " steps_plus_1"],
    conditioning: str,
    batch_size: int,
    cond_images: Optional[Float[Array, "batch 1 h w"]] = None,
    labels: Optional[Int[Array, " batch"]] = None,
    image_shape: tuple = (1, 28, 28),
) -> Float[Array, "batch c h w"]:
    """
    same as sample_batch_x except it understands conditioning
    cond_images is the set of real images used for the lowres/known-region cases
    labels are the digit labels for class conditioning
    conditioning has 4 options:
    none - cond_images and labels both not needed
    class - needs labels
    lowres - needs cond_images
    inpaint - needs cond_images
    """
    if conditioning in ("lowres", "inpaint"):
        assert (
            cond_images is not None
        ), f"conditioning={conditioning} requires cond_images"
    if conditioning == "class":
        assert labels is not None, "conditioning=class requires labels"

    key_noise, key_fixed_noise = jax.random.split(key)
    z = jax.random.normal(key_noise, (batch_size,) + image_shape)

    extra = (
        build_cond_channels(conditioning, cond_images)
        if cond_images is not None
        else None
    )
    # fixed noise sample for the known region
    fixed_noise = (
        jax.random.normal(key_fixed_noise, (batch_size,) + image_shape)
        if conditioning == "inpaint"
        else None
    )

    num_steps = len(timesteps) - 1

    for i in range(num_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        t_batched = jnp.full(batch_size, t)

        if conditioning == "inpaint":
            t_full = jnp.full((batch_size, 1, 1, 1), t)
            z = inject_known(conditioning, z, cond_images, fixed_noise, t_full)

        model_input = z if extra is None else jnp.concatenate([z, extra], axis=1)

        if labels is not None:
            x_pred = jax.vmap(model)(model_input, t_batched, labels)
        else:
            x_pred = jax.vmap(model)(model_input, t_batched)

        v = (x_pred - z) / jnp.maximum(1 - t, 0.05)
        z = z + (t_next - t) * v

    if conditioning == "inpaint":
        t_full = jnp.full((batch_size, 1, 1, 1), timesteps[-1])
        z = inject_known(conditioning, z, cond_images, fixed_noise, t_full)

    return z
