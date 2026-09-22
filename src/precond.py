"""EDM preconditioning (Karras et al. 2022) for this repo's interpolant.

The paper's image experiments use it verbatim -- Appendix C.2: "The denoiser
uses the EDM preconditioning coefficients c_in, c_skip, c_out, c_noise with
sigma_data = 0.5, sigma_min = 0.002, and sigma_max = 80." We did not, and that
is the leading candidate for two measured defects:

  * m_hat is flat over the lowest bins where any valid mmse falls as sigma^2.
    With preconditioning, c_skip -> 1 and c_out -> 0 as sigma -> 0, so the
    denoiser returns its input and the error goes as sigma^2 automatically,
    without the network having to learn to copy.
  * the fixed loss weight w = 1/max(t_clip, 1-t)^2 diverges as 1/sigma^2 with
    nothing to cancel it. EDM's weight has the same shape but equals
    1/c_out(sigma)^2 exactly, so it cancels c_out and leaves the network a
    uniform objective at every noise level (see src/loss.py).

Coordinates. This repo interpolates z = t*x0 + (1-t)*eps, which factors as

    z = t * (x0 + sigma*eps),        sigma = (1 - t) / t

so y = z / t is exactly the VE variable EDM is written for. No approximation is
involved; the same sigma = (1-t)/t relation src/infonoise.py already uses.

    D(z, t) = c_skip(sigma)*y + c_out(sigma)*F(c_in(sigma)*y, c_noise(sigma))

sigma_data stays at the paper's 0.5 by default and is NOT scaled per dataset,
which is deliberate. If sigma_data scaled with the data then at sigma' = k*sigma
every coefficient would rescale so that F's input and target are identical to
k=1 (differing only by a constant ln(k)/4 in c_noise) -- training at k=10 would
be a relabelled copy of k=1 and the data-scale axis would measure nothing. Fixed
at 0.5 the shift stays real, which also matches how the paper applies its image
setup to domains it was not calibrated for. Pass --sigma_data to override; the
scaled setting is useful as a correctness control, where k=1 and k=10 must give
identical results.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Appendix C.2. Not scaled per dataset -- see the module docstring.
DEFAULT_SIGMA_DATA = 0.5


def sigma_of_t(t):
    """Noise-to-signal ratio for z = t*x0 + (1-t)*eps, matching src.infonoise."""
    return (1.0 - t) / jnp.maximum(t, 1e-12)


def coefficients(sigma, sigma_data: float = DEFAULT_SIGMA_DATA):
    """(c_skip, c_out, c_in, c_noise), EDM Table 1."""
    ss = sigma**2 + sigma_data**2
    c_skip = sigma_data**2 / ss
    c_out = sigma * sigma_data / jnp.sqrt(ss)
    c_in = 1.0 / jnp.sqrt(ss)
    c_noise = jnp.log(jnp.maximum(sigma, 1e-12)) / 4.0
    return c_skip, c_out, c_in, c_noise


def loss_weight_edm(sigma, sigma_data: float = DEFAULT_SIGMA_DATA):
    """w(sigma) = 1/c_out^2, EDM Eq. 8 and this paper's Eq. 38.

    Quoted in the paper as (sigma^2 + sigma_data^2)/(sigma^2 sigma_data^2),
    which is the same thing. It is what makes the x-space objective reduce to
    plain MSE on F's own output.
    """
    return (sigma**2 + sigma_data**2) / (sigma**2 * sigma_data**2)


def network_target(
    clean: Float[Array, "b c h w"],
    z: Float[Array, "b c h w"],
    t: Float[Array, "b 1 1 1"],
    sigma_data: float = DEFAULT_SIGMA_DATA,
):
    """What F should predict, and the input it should predict it from.

    Returns (model_input, target, c_noise). Multiplying EDM's weight 1/c_out^2
    through ||D - x0||^2 cancels c_out and leaves ||F - target||^2, so the
    training objective is plain MSE on this target and no loss weight or
    t_clip appears anywhere.

    The target is finite at sigma -> 0 despite c_out -> 0: with y = x0 + sigma*eps
    it simplifies to (x0*sigma - sigma_data^2*eps) / (sqrt(sigma^2 + sigma_data^2)
    * sigma_data), which tends to -eps. The 1e-12 floor below is for sigma
    exactly 0, which the samplers never pass.
    """
    sigma = sigma_of_t(t)
    c_skip, c_out, c_in, c_noise = coefficients(sigma, sigma_data)
    y = z / jnp.maximum(t, 1e-12)
    model_input = c_in * y
    target = (clean - c_skip * y) / jnp.maximum(c_out, 1e-12)
    return model_input, target, c_noise


def denoise(
    model,
    z: Float[Array, "b c h w"],
    t: Float[Array, "b 1 1 1"],
    sigma_data: float = DEFAULT_SIGMA_DATA,
    extra=None,
    labels=None,
) -> Float[Array, "b c h w"]:
    """D(z, t): the preconditioned prediction of the clean image.

    Drop-in for `jax.vmap(model)(z, t)` wherever an x-prediction is wanted --
    training, sampling, and evaluation all go through this so they cannot drift
    apart. ``extra`` is the conditioning channels, concatenated to the SCALED
    input so the network sees them at their own magnitude rather than through
    c_in.
    """
    sigma = sigma_of_t(t)
    c_skip, c_out, c_in, c_noise = coefficients(sigma, sigma_data)
    y = z / jnp.maximum(t, 1e-12)
    model_input = c_in * y
    if extra is not None:
        model_input = jnp.concatenate([model_input, extra], axis=1)

    flat_noise = c_noise.reshape(-1)
    if labels is not None:
        f = jax.vmap(model)(model_input, flat_noise, labels)
    else:
        f = jax.vmap(model)(model_input, flat_noise)
    return c_skip * y + c_out * f
