from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from src.conditioning import build_cond_channels

# def compute_loss_jit(model: eqx.Module, clean_images: Float[Array, "b c h w"], noise: Float[Array, "b c h w"], t: Float[Array, "b 1 1 1"], t_clip: float = 0.05) -> Float[Array, ""]:
#     z = t * clean_images + (1 - t) * noise
#     x_pred = jax.vmap(model)(z, t.squeeze())
#     v_pred = (x_pred - z) / jnp.maximum(1 - t, t_clip)
#     v_true = (clean_images - z) / jnp.maximum(1 - t, t_clip)
#     return jnp.mean((v_pred - v_true) ** 2)


def compute_loss_x(
    model: eqx.Module,
    clean_images: Float[Array, "b c h w"],
    noise: Float[Array, "b c h w"],
    t: Float[Array, "b 1 1 1"],
) -> Float[Array, ""]:
    z = t * clean_images + (1 - t) * noise
    x_pred = jax.vmap(model)(z, t.reshape(-1))
    return jnp.mean((x_pred - clean_images) ** 2 / jnp.maximum(0.05, 1 - t) ** 2)


def compute_loss_cond(
    model: eqx.Module,
    conditioning: str,
    clean_images: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
    labels: Optional[Int[Array, " b"]] = None,
) -> Float[Array, ""]:
    """
    same x-pred loss as in compute_loss_x above
    concatenates conditioning variant's extra channels if they exist onto noisy image
    labels get passed through
    loss is over full image
    """
    z = t * clean_images + (1 - t) * noise
    extra = build_cond_channels(conditioning, clean_images)
    model_input = z if extra is None else jnp.concatenate([z, extra], axis=1)

    if labels is not None:
        x_pred = jax.vmap(model)(model_input, t.reshape(-1), labels)
    else:
        x_pred = jax.vmap(model)(model_input, t.reshape(-1))

    return jnp.mean((x_pred - clean_images) ** 2 / jnp.maximum(0.05, 1 - t) ** 2)
