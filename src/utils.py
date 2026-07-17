import os
import ssl
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import requests
import tensorflow_datasets as tfds
import tqdm
from jaxtyping import Array, Float, Int, PRNGKeyArray
from PIL import Image

from src.conditioning import build_cond_channels, inject_known


def get_mnist_dataloaders(batch_size: int, with_labels: bool = False):
    """
    load and pre-process MNSIT data
    if with_labels is true then returnthe digit label array too (need it for class conditioning)
    """

    data_dir = os.path.join("data", "tensorflow_datasets")
    ds = tfds.load("mnist", split="train", as_supervised=True, data_dir=data_dir)
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        images.append(img)
        labels.append(label)

    images = np.stack(images)
    images = images / 127.5 - 1.0  # normalize to [-1, 1]

    # convert from (N, H, W, C) to (N, C, H, W)
    images = images.transpose(0, 3, 1, 2)
    images = jnp.array(images)

    if not with_labels:
        return images

    labels = jnp.array(np.stack(labels), dtype=jnp.int32)
    return images, labels


# luminance weights for RGB -> grayscale. Plain tuple, not a jnp.array, so
# importing this module never forces JAX to initialize its GPU/cuDNN backend
# as a side effect -- that should only happen once train.py's own "no GPU
# visible" check has had a chance to run and fail cleanly.
_GRAY_WEIGHTS = (0.299, 0.587, 0.114)

# madm.dfki.de (EuroSAT's host) sends a misconfigured intermediate chain --
# the leaf cert is issued by "HARICA GEANT TLS RSA 1", but the extra certs
# the server sends are for an unrelated chain, so plain cert verification
# fails with CERTIFICATE_VERIFY_FAILED even though the *root* (HARICA TLS RSA
# Root CA 2021) is already in certifi's bundle. Fetch the one missing
# intermediate from its own AIA URL (embedded in the leaf cert) and layer it
# onto certifi's bundle -- this fixes verification without disabling it.
_HARICA_GEANT_INTERMEDIATE_URL = "http://crt.harica.gr/HARICA-GEANT-TLS-R1.cer"


def _eurosat_ca_bundle() -> str:
    import certifi

    cache_path = os.path.join(
        "data", "tensorflow_datasets", ".eurosat_ca_bundle.pem"
    )
    if os.path.exists(cache_path):
        return cache_path

    intermediate_der = requests.get(_HARICA_GEANT_INTERMEDIATE_URL, timeout=15).content
    intermediate_pem = ssl.DER_cert_to_PEM_cert(intermediate_der)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(certifi.where()) as f:
        bundle = f.read()
    with open(cache_path, "w") as f:
        f.write(bundle + "\n" + intermediate_pem)
    return cache_path


def get_eurosat_dataloaders(batch_size: int, with_labels: bool = False):
    """
    load and pre-process EuroSAT (RGB variant) to look like MNIST input:
    grayscale, resize 64->32, then center-crop 28x28
    """

    # tfds's registered eurosat download URL is still plain http://, which the
    # host now 403s (https:// works fine) -- patch it in-place before loading.
    import tensorflow_datasets.image_classification.eurosat as _eurosat_module

    for _cfg in _eurosat_module.Eurosat.BUILDER_CONFIGS:
        _cfg.download_url = _cfg.download_url.replace("http://", "https://")

    # requests (used internally by tfds's downloader) honors this env var for
    # the CA bundle -- see _eurosat_ca_bundle's docstring/comment above.
    try:
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _eurosat_ca_bundle())
    except requests.RequestException:
        pass  # fall back to default verification; download will just fail loudly

    data_dir = os.path.join("data", "tensorflow_datasets")
    ds = tfds.load("eurosat/rgb", split="train", as_supervised=True, data_dir=data_dir)
    print("Decoding eurosat examples...")
    images = []
    labels = []
    for img, label in tqdm.tqdm(tfds.as_numpy(ds), total=27000, unit="img"):
        images.append(img)
        labels.append(label)

    print("Stacking into one array...")
    images = np.stack(images).astype(np.float32)  # (N, 64, 64, 3)
    images = jnp.array(images)

    # grayscale via luminance weights -> (N, 64, 64, 1)
    print("Grayscaling + resizing 64->32 + cropping to 28x28...")
    images = jnp.sum(images * jnp.array(_GRAY_WEIGHTS), axis=-1, keepdims=True)

    # resize 64 -> 32
    n = images.shape[0]
    images = jax.image.resize(images, (n, 32, 32, 1), method="bilinear")

    # center-crop 32 -> 28
    offset = (32 - 28) // 2
    images = images[:, offset : offset + 28, offset : offset + 28, :]

    images = images / 127.5 - 1.0  # normalize to [-1, 1]

    # convert from (N, H, W, C) to (N, C, H, W)
    images = images.transpose(0, 3, 1, 2)

    if not with_labels:
        return images

    labels = jnp.array(np.stack(labels), dtype=jnp.int32)
    return images, labels


_DATASET_LOADERS = {
    "mnist": get_mnist_dataloaders,
    "eurosat": get_eurosat_dataloaders,
}


def get_dataloaders(dataset: str, batch_size: int, with_labels: bool = False):
    """dispatch to the right per-dataset loader (mnist, eurosat)"""
    return _DATASET_LOADERS[dataset](batch_size=batch_size, with_labels=with_labels)


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
    cond_images: Optional[Float[Array, "batch 1 28 28"]] = None,
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
