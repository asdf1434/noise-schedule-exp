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


def inpaint_mask(height: int = 28, width: int = 28) -> Bool[Array, "1 height width"]:
    """
    true means pixel is given to the model
    false means it has to be generated
    in this exmple the true is in the left half
    """
    mask = jnp.zeros((1, height, width), dtype=bool)
    mask = mask.at[:, :, : width // 2].set(True)
    return mask


def make_inpaint_channels(
    clean: Float[Array, "b 1 h w"],
) -> Float[Array, "b 2 h w"]:
    """
    builds 2 extra conditioning channels for in-painting
    this includes the mask, and the actual real values
    """
    mask_f = inpaint_mask(clean.shape[2], clean.shape[3]).astype(clean.dtype)
    known = clean * mask_f
    known_batched = jnp.broadcast_to(known, clean.shape)
    mask_batched = jnp.broadcast_to(mask_f, clean.shape)
    return jnp.concatenate([known_batched, mask_batched], axis=1)


def inject_known_region(
    z: Float[Array, "b 1 h w"],
    clean: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
) -> Float[Array, "b 1 h w"]:
    """
    this overwrites the known half of z to be the true values
    this is needed because every step you re-write the real values into the generation
    """
    mask = inpaint_mask(clean.shape[2], clean.shape[3]).astype(clean.dtype)
    known_z = t * clean + (1 - t) * noise
    return jnp.where(mask, known_z, z)


# actually use in one function


def build_cond_channels(
    name: str, clean: Float[Array, "b 1 h w"]
) -> Optional[Float[Array, "b c h w"]]:
    """
    pick the right option
    """
    if name == "lowres":
        return make_lowres_channel(clean)
    if name == "inpaint":
        return make_inpaint_channels(clean)
    return None


def inject_known(
    name: str,
    z: Float[Array, "b 1 h w"],
    clean: Float[Array, "b 1 h w"],
    noise: Float[Array, "b 1 h w"],
    t: Float[Array, "b 1 1 1"],
) -> Float[Array, "b 1 h w"]:
    if name == "inpaint":
        return inject_known_region(z, clean, noise, t)
    return z
