from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from src.conditioning import build_cond_channels

# Floor on (1 - t) in the x-prediction loss weight 1/max(T_CLIP, 1-t)^2, which
# would otherwise diverge as t -> 1. src/infonoise.py imports this so its
# w(sigma) is the same fixed weight the objective actually uses.
T_CLIP = 0.05

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
    return jnp.mean((x_pred - clean_images) ** 2 / jnp.maximum(T_CLIP, 1 - t) ** 2)


def compute_loss_cond(
    model: eqx.Module,
    conditioning: str,
    clean_images: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
    labels: Optional[Int[Array, " b"]] = None,
    cond_params=(),
) -> tuple[Float[Array, ""], Float[Array, " b"]]:
    """
    same x-pred loss as in compute_loss_x above
    concatenates conditioning variant's extra channels if they exist onto noisy image
    labels get passed through
    loss is over full image
    cond_params tunes how much help the conditioning gives -- see CONDITIONING_PARAMS

    Returns (weighted scalar loss used for the gradient, per-sample unweighted
    squared error used only as an InfoNoise profile statistic).
    """
    z = t * clean_images + (1 - t) * noise
    extra = build_cond_channels(conditioning, clean_images, cond_params)
    model_input = z if extra is None else jnp.concatenate([z, extra], axis=1)

    if labels is not None:
        x_pred = jax.vmap(model)(model_input, t.reshape(-1), labels)
    else:
        x_pred = jax.vmap(model)(model_input, t.reshape(-1))

    sq_err = (x_pred - clean_images) ** 2
    weighted = jnp.mean(sq_err / jnp.maximum(T_CLIP, 1 - t) ** 2)
    # Per-sample *unweighted* denoising error, i.e. an online sample of
    # mmse(t). Unused by the objective; InfoNoise (src/infonoise.py) bins it to
    # estimate the information profile. Free to compute -- it is the same
    # residual the weighted loss already forms.
    unweighted = jnp.mean(sq_err, axis=(1, 2, 3))
    return weighted, unweighted
