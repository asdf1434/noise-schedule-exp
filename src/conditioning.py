"""
contains conditioning variants for question 2

defines all of them in the same place for convenience
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

# big table of how conditioning works and what args are required for each type
CONDITIONING = {
    "none": {"in_channels": 1, "num_classes": None, "needs_labels": False},
    "class": {"in_channels": 1, "num_classes": 10, "needs_labels": True},
    "lowres": {"in_channels": 2, "num_classes": None, "needs_labels": False},
    "inpaint": {"in_channels": 3, "num_classes": None, "needs_labels": False},
}

# lowres


def make_lowres_channel(
    clean: Float[Array, "b 1 h w"], factor: int = 4
) -> Float[Array, "b 1 h w"]:
    """
    downsapmle by factor on each dim, and then upscale using nearest neighbor
    u end up with a block version of the image
    """
    b, c, h, w = clean.shape
    lo_h, lo_w = h // factor, w // factor
    lo = jax.image.resize(clean, (b, c, lo_h, lo_w), method="linear")
    return jax.image.resize(lo, (b, c, h, w), method="nearest")


# inpaint


def inpaint_mask(
    height: int = 28, width: int = 28, known_fraction: float = 0.5
) -> Bool[Array, "1 height width"]:
    """
    true means pixel is given to the model
    false means it has to be generated
    the true region is the leftmost known_fraction of the columns
    (0.5 -- the left half -- is the original behaviour)
    """
    mask = jnp.zeros((1, height, width), dtype=bool)
    mask = mask.at[:, :, : int(round(width * known_fraction))].set(True)
    return mask


def make_inpaint_channels(
    clean: Float[Array, "b 1 h w"], known_fraction: float = 0.5
) -> Float[Array, "b 2 h w"]:
    """
    builds 2 extra conditioning channels for in-painting
    this includes the mask, and the actual real values
    """
    mask_f = inpaint_mask(clean.shape[2], clean.shape[3], known_fraction).astype(
        clean.dtype
    )
    known = clean * mask_f
    known_batched = jnp.broadcast_to(known, clean.shape)
    mask_batched = jnp.broadcast_to(mask_f, clean.shape)
    return jnp.concatenate([known_batched, mask_batched], axis=1)


def inject_known_region(
    z: Float[Array, "b 1 h w"],
    clean: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
    known_fraction: float = 0.5,
) -> Float[Array, "b 1 h w"]:
    """
    this overwrites the known region of z to be the true values
    this is needed because every step you re-write the real values into the generation
    """
    mask = inpaint_mask(clean.shape[2], clean.shape[3], known_fraction).astype(
        clean.dtype
    )
    known_z = t * clean + (1 - t) * noise
    return jnp.where(mask, known_z, z)


# actually use in one function

# How much help each variant gives is tunable, so the substitution effect can be
# measured as a dose-response curve rather than a single point. Params arrive as
# a tuple of (key, value) pairs rather than a dict because these functions are
# called from inside jitted code, where arguments have to be hashable -- and
# `factor` in particular decides an array shape, so it can't be a traced value.
CONDITIONING_PARAMS = {
    "lowres": ("factor", 4),
    "inpaint": ("known_fraction", 0.5),
}


def _param(name: str, params) -> float:
    """The single tunable for this variant, or its default if unset."""
    key, default = CONDITIONING_PARAMS[name]
    return dict(params or ()).get(key, default)


def build_cond_channels(
    name: str, clean: Float[Array, "b 1 h w"], params=()
) -> Optional[Float[Array, "b c h w"]]:
    """
    pick the right option
    """
    if name == "lowres":
        return make_lowres_channel(clean, int(_param(name, params)))
    if name == "inpaint":
        return make_inpaint_channels(clean, float(_param(name, params)))
    return None


def inject_known(
    name: str,
    z: Float[Array, "b 1 h w"],
    clean: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
    params=(),
) -> Float[Array, "b 1 h w"]:
    if name == "inpaint":
        return inject_known_region(z, clean, noise, t, float(_param(name, params)))
    return z
