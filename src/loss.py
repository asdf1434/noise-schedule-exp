from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from src.conditioning import build_cond_channels

# Default floor on (1 - t) in the x-prediction loss weight 1/max(t_clip, 1-t)^2,
# which would otherwise diverge as t -> 1. src/infonoise.py imports this so its
# w(sigma) is the same fixed weight the objective actually uses.
#
# The floor is a dead zone: for 1 - t <= t_clip the weight stops growing and the
# sampler's velocity is understated. In sigma = (1-t)/t coordinates that is
# sigma <= t_clip/(1 - t_clip), i.e. sigma <= 0.0526 at the default.
#
# That default assumes data at roughly unit amplitude. On a dataset scaled down
# by k (DatasetSpec.sample_scale), the informative sigma range translates down
# by k and moves INTO the dead zone -- measured on mnist_x0.1, 51% of the
# training mass lands at sigma <= 0.0526, versus 8.9% on mnist_x10. So scaled
# runs need t_clip scaled with them, which is what --t_clip is for.
#
# Scaling UP is not symmetric and not wanted: t_clip = 0.5 would clip everything
# below sigma = 1. Only the k < 1 direction needs adjusting.
T_CLIP = 0.05

# Named loss weightings, in sigma = (1-t)/t coordinates:
#   "vpred"   w(sigma) = 1/max(T_CLIP, 1-t)^2, this repo's default and the
#             x-prediction equivalent of v-prediction weighting (david.md).
#   "uniform" w == 1, i.e. plain unweighted x-space MSE.
# The weighting matters to more than the gradient: the effective allocation a
# training run sees is pi(sigma)*w(sigma), and src/infonoise.py divides w back
# out when it builds pi (Eq. 15). With "vpred", w varies by ~100x across the
# useful sigma range and so dominates where training effort lands whatever pi
# does; with "uniform" the training distribution IS the allocation. Anything
# comparing training distributions on FID has to hold this fixed.
LOSS_WEIGHTINGS = ("vpred", "uniform")
DEFAULT_LOSS_WEIGHTING = "vpred"


def weight_of_t(
    t, loss_weighting: str = DEFAULT_LOSS_WEIGHTING, t_clip: float = T_CLIP
):
    """Per-sample loss weight for the given weighting, in t coordinates."""
    if loss_weighting == "uniform":
        return jnp.ones_like(t)
    if loss_weighting == "vpred":
        return 1.0 / jnp.maximum(t_clip, 1 - t) ** 2
    raise ValueError(
        f"unknown loss_weighting {loss_weighting!r}; expected one of {LOSS_WEIGHTINGS}"
    )

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
    loss_weighting: str = DEFAULT_LOSS_WEIGHTING,
    t_clip: float = T_CLIP,
) -> Float[Array, ""]:
    z = t * clean_images + (1 - t) * noise
    x_pred = jax.vmap(model)(z, t.reshape(-1))
    return jnp.mean(
        (x_pred - clean_images) ** 2 * weight_of_t(t, loss_weighting, t_clip)
    )


def compute_loss_cond(
    model: eqx.Module,
    conditioning: str,
    clean_images: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
    labels: Optional[Int[Array, " b"]] = None,
    cond_params=(),
    loss_weighting: str = DEFAULT_LOSS_WEIGHTING,
    t_clip: float = T_CLIP,
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
    weighted = jnp.mean(sq_err * weight_of_t(t, loss_weighting, t_clip))
    # Per-sample *unweighted* denoising error, i.e. an online sample of
    # mmse(t). Unused by the objective; InfoNoise (src/infonoise.py) bins it to
    # estimate the information profile. Free to compute -- it is the same
    # residual the weighted loss already forms.
    unweighted = jnp.mean(sq_err, axis=(1, 2, 3))
    return weighted, unweighted
