"""Dataset registry -- the one place that knows about each dataset's geometry
and how to load it.

Adding a new dataset = add one ``DatasetSpec`` entry to ``DATASETS`` (and, for
the common "grayscale + resize to a square target" case, you can reuse
``load_tfds_dataset`` so the entry is only a few lines). Everything downstream
-- the model's input shape, train.py's batching reshape, the sampler's
``image_shape``, and the FID real-image directory / cached-stats name -- is
derived from the spec instead of being hardcoded to 28x28 MNIST, so images are
no longer assumed to be 28x28.
"""

import os
import ssl
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import requests
import tensorflow_datasets as tfds
import tqdm

# luminance weights for RGB -> grayscale. Plain tuple, not a jnp.array, so
# importing this module never forces JAX to initialize its GPU/cuDNN backend
# as a side effect -- that should only happen once train.py's own "no GPU
# visible" check has had a chance to run and fail cleanly.
_GRAY_WEIGHTS = (0.299, 0.587, 0.114)

_DATA_DIR = os.path.join("data", "tensorflow_datasets")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know about a dataset.

    ``load(batch_size, with_labels)`` returns images ``(N, C, H, W)`` in
    ``[-1, 1]`` (and a parallel label array when ``with_labels`` is True),
    exactly like the old per-dataset loaders did.
    """

    name: str
    image_size: int  # H == W, in pixels (e.g. 28 for mnist, 64 for eurosat64)
    channels: int  # C -- currently 1 (grayscale) everywhere
    num_classes: Optional[int]  # for class conditioning / reference only
    real_dir: str  # where generate_real_samples dumps + FID reads real images
    real_stats_name: str  # clean-fid cached-stats key (see cache_real_stats.py)
    load: Callable[..., object]


def _finalize(images_nhwc: np.ndarray) -> jnp.ndarray:
    """Normalize [0,255] -> [-1,1] and convert NHWC -> NCHW."""
    images = images_nhwc / 127.5 - 1.0
    images = images.transpose(0, 3, 1, 2)
    return jnp.asarray(images)


def _grayscale(images_nhwc: jnp.ndarray) -> jnp.ndarray:
    """Collapse an RGB (…,3) batch to a single luminance channel (…,1).

    A no-op if the batch is already single-channel.
    """
    if images_nhwc.shape[-1] == 1:
        return images_nhwc
    return jnp.sum(images_nhwc * jnp.array(_GRAY_WEIGHTS), axis=-1, keepdims=True)


def load_tfds_dataset(
    tfds_name: str,
    target_size: int,
    grayscale: bool = False,
    total: Optional[int] = None,
    with_labels: bool = False,
):
    """Generic loader for the common case: load a tfds image classification
    dataset, optionally grayscale it, resize to ``target_size`` x
    ``target_size``, and normalize to the model's ``[-1, 1]`` NCHW format.

    Datasets with quirks (custom download URLs, TLS chains, crops) get their
    own loader function instead -- see ``get_eurosat_dataloaders`` -- but the
    majority can be registered with just a partial over this.
    """
    ds = tfds.load(tfds_name, split="train", as_supervised=True, data_dir=_DATA_DIR)
    images, labels = [], []
    for img, label in tqdm.tqdm(tfds.as_numpy(ds), total=total, unit="img"):
        images.append(img)
        labels.append(label)

    images = jnp.asarray(np.stack(images).astype(np.float32))  # (N, H, W, C)
    if grayscale:
        images = _grayscale(images)

    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    if (h, w) != (target_size, target_size):
        images = jax.image.resize(
            images, (n, target_size, target_size, images.shape[-1]), method="bilinear"
        )

    images = _finalize(np.asarray(images))
    if not with_labels:
        return images
    return images, jnp.array(np.stack(labels), dtype=jnp.int32)


def get_mnist_dataloaders(batch_size: int, with_labels: bool = False):
    """Load MNIST (native 28x28 grayscale) in the model's NCHW [-1,1] format."""
    return load_tfds_dataset(
        "mnist", target_size=28, grayscale=False, total=60000, with_labels=with_labels
    )


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

    cache_path = os.path.join(_DATA_DIR, ".eurosat_ca_bundle.pem")
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


def get_eurosat_dataloaders(
    batch_size: int, with_labels: bool = False, target_size: int = 28
):
    """Load EuroSAT (RGB variant, native 64x64) as grayscale in the model's
    NCHW [-1,1] format.

    ``target_size=28`` reproduces the original MNIST-matched pipeline (resize
    64->32 then center-crop to 28x28) so existing 28x28 EuroSAT results stay
    comparable. Any other ``target_size`` just resizes 64 -> target_size with
    no crop (e.g. ``target_size=64`` keeps the native resolution).
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

    ds = tfds.load(
        "eurosat/rgb", split="train", as_supervised=True, data_dir=_DATA_DIR
    )
    print("Decoding eurosat examples...")
    images, labels = [], []
    for img, label in tqdm.tqdm(tfds.as_numpy(ds), total=27000, unit="img"):
        images.append(img)
        labels.append(label)

    print("Stacking into one array...")
    images = jnp.asarray(np.stack(images).astype(np.float32))  # (N, 64, 64, 3)
    images = _grayscale(images)  # -> (N, 64, 64, 1)
    n = images.shape[0]

    if target_size == 28:
        print("Resizing 64->32 + cropping to 28x28...")
        images = jax.image.resize(images, (n, 32, 32, 1), method="bilinear")
        offset = (32 - 28) // 2  # center-crop 32 -> 28
        images = images[:, offset : offset + 28, offset : offset + 28, :]
    else:
        print(f"Resizing 64->{target_size}...")
        if target_size != 64:
            images = jax.image.resize(
                images, (n, target_size, target_size, 1), method="bilinear"
            )

    images = _finalize(np.asarray(images))
    if not with_labels:
        return images
    return images, jnp.array(np.stack(labels), dtype=jnp.int32)


def _eurosat64_loader(batch_size: int, with_labels: bool = False):
    return get_eurosat_dataloaders(batch_size, with_labels, target_size=64)


def get_fashion_mnist_dataloaders(batch_size: int, with_labels: bool = False):
    """Load Fashion-MNIST (native 28x28 grayscale) in the model's NCHW [-1,1] format."""
    return load_tfds_dataset(
        "fashion_mnist",
        target_size=28,
        grayscale=False,
        total=60000,
        with_labels=with_labels,
    )


def get_cifar10_dataloaders(batch_size: int, with_labels: bool = False):
    """Load CIFAR-10 (native 32x32 RGB) collapsed to grayscale, matching the
    treatment EuroSAT gets -- keeps the model/conditioning/FID pipeline's
    single-channel assumption intact instead of adding RGB support.

    Pulled from HuggingFace Hub (`uoft-cs/cifar10`, parquet, CDN-backed)
    instead of tfds's `cifar10` builder -- tfds's builder downloads the
    binary-format tarball straight from cs.toronto.edu, a plain university
    server with no CDN that was measured at ~1.7MB/min (~100min for the full
    ~163MB file); the HF mirror downloaded the same data in seconds.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "uoft-cs/cifar10", split="train", cache_dir=os.path.join(_DATA_DIR, "hf_cache")
    )
    images, labels = [], []
    for ex in tqdm.tqdm(ds, total=len(ds), unit="img"):
        images.append(np.array(ex["img"], dtype=np.float32))
        labels.append(ex["label"])

    images = jnp.asarray(np.stack(images))  # (N, 32, 32, 3)
    images = _grayscale(images)
    images = _finalize(np.asarray(images))
    if not with_labels:
        return images
    return images, jnp.array(np.stack(labels), dtype=jnp.int32)


DATASETS = {
    "mnist": DatasetSpec(
        name="mnist",
        image_size=28,
        channels=1,
        num_classes=10,
        real_dir="data/real",
        real_stats_name="mnist_real",
        load=get_mnist_dataloaders,
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        image_size=28,
        channels=1,
        num_classes=10,
        real_dir="data/real_fashion_mnist",
        real_stats_name="fashion_mnist_real",
        load=get_fashion_mnist_dataloaders,
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        image_size=32,
        channels=1,
        num_classes=10,
        real_dir="data/real_cifar10",
        real_stats_name="cifar10_real",
        load=get_cifar10_dataloaders,
    ),
    "eurosat": DatasetSpec(
        name="eurosat",
        image_size=28,
        channels=1,
        num_classes=10,
        real_dir="data/real_eurosat",
        real_stats_name="eurosat_real",
        load=get_eurosat_dataloaders,
    ),
    "eurosat64": DatasetSpec(
        name="eurosat64",
        image_size=64,
        channels=1,
        num_classes=10,
        real_dir="data/real_eurosat64",
        real_stats_name="eurosat64_real",
        load=_eurosat64_loader,
    ),
}


def get_dataloaders(dataset: str, batch_size: int, with_labels: bool = False):
    """Dispatch to the right per-dataset loader via the registry."""
    return DATASETS[dataset].load(batch_size=batch_size, with_labels=with_labels)
