import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

# training distributions


def sample_t_uniform(key: PRNGKeyArray, batch_size: int) -> Float[Array, "batch 1 1 1"]:
    t = jax.random.uniform(key, (batch_size,))
    return t.reshape(-1, 1, 1, 1)


def sample_t_logit_normal(
    key: PRNGKeyArray, batch_size: int, mu: float = 0.0, sigma: float = 1.0
) -> Float[Array, "batch 1 1 1"]:
    normal_samples = jax.random.normal(key, (batch_size,))
    t = jax.nn.sigmoid(mu + sigma * normal_samples)
    return t.reshape(-1, 1, 1, 1)


def sample_t_plateau_logit_normal(
    key: PRNGKeyArray,
    batch_size: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    uniform_prob: float = 0.5,
) -> Float[Array, " batch 1 1 1"]:
    """
    sample from plateau-logit-normal distribution
    """
    # need 3 keys since 3 random operations
    key_ln, key_unif, key_mask = jax.random.split(key, 3)

    # sample logit-normal
    z = jax.random.normal(key_ln, (batch_size,)) * sigma + mu
    t_ln = jax.nn.sigmoid(z)

    # sample uniform
    t_unif = jax.random.uniform(key_unif, (batch_size,))

    # this is a mask to decide which one gets logit-normal, which gets uniform
    mask = jax.random.bernoulli(key_mask, p=uniform_prob, shape=(batch_size,))
    t = jnp.where(mask, t_unif, t_ln)

    return t.reshape(-1, 1, 1, 1)


# inference sequences


def get_uniform_steps(num_steps: int) -> Float[Array, " num_steps_plus_1"]:
    return jnp.linspace(0.0, 1.0, num_steps + 1)


def get_shifted_steps(
    num_steps: int, shift: float = 0.5
) -> Float[Array, " num_steps_plus_1"]:
    # cluster steps based on shift
    u = jnp.linspace(0.0, 1.0, num_steps + 1)
    t = (u * shift) / (1 + (shift - 1) * u)
    return t


def get_logit_normal_cdf_steps(
    num_steps: int, mu: float = 0.0, sigma: float = 1.0
) -> Float[Array, " num_steps_plus_1"]:
    # cluster steps based on logit-normal distribution cdf

    eps = 1e-5
    p = jnp.linspace(eps, 1.0 - eps, num_steps + 1)
    # inverse cdf of normal
    normal_quantiles = jax.scipy.special.ndtri(p)
    # transform to logit-normal
    t = jax.nn.sigmoid(mu + sigma * normal_quantiles)
    return jnp.clip(t, 0.0, 1.0)
