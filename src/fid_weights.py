"""Race-free staging of clean-fid's InceptionV3 weights.

cleanfid hardcodes its TorchScript weight cache to ``/tmp`` (see
``cleanfid/features.py``), and downloads into it like this::

    if not os.path.exists(inception_path):
        with urllib.request.urlopen(url) as r, open(inception_path, 'wb') as f:
            shutil.copyfileobj(r, f)

No lock, no temp file, no atomic rename. ``/tmp`` is node-local, so when a
Slurm array puts several eval tasks on ONE node they race: the first opener
creates the file, every other task then sees ``os.path.exists`` return True,
skips the download, and ``torch.jit.load``s a file that is still streaming.
That surfaces as::

    RuntimeError: PytorchStreamReader failed reading zip archive:
                  failed finding central directory

It is also sticky. The truncated file stays on that node's ``/tmp`` forever,
and since the check is existence-only cleanfid never re-downloads it, so every
later task landing there fails deterministically. This cost 10 of 64 tasks in
job 1523740.

The fix is two staged copies:

1. Download once to a SHARED, repo-local cache (``data/fid_cache/``) using
   write-to-temp + ``os.replace``, so a partial download is never visible
   under the real name even if two jobs start together.
2. Copy that into the node's ``/tmp`` the same atomic way, after checking
   whatever is already there actually loads.

Both steps are idempotent and safe to run concurrently, which means the eval
array no longer needs the prep stage to have primed anything -- though calling
this from prep still avoids N nodes each pulling ~91 MB from NVIDIA's CDN.
"""

import os
import shutil
import urllib.request
from pathlib import Path

import torch

# Must match cleanfid.downloads_helper.inception_url.
INCEPTION_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
    "pretrained/metrics/inception-2015-12-05.pt"
)
WEIGHT_NAME = "inception-2015-12-05.pt"

# cleanfid looks here and nowhere else (non-Windows).
CLEANFID_CACHE_DIR = Path("/tmp")

# Shared filesystem, so the download happens once per checkout rather than
# once per node. data/ is gitignored.
SHARED_CACHE_DIR = Path("data/fid_cache")


def _loads_cleanly(path: Path) -> bool:
    """True if `path` is a complete TorchScript archive.

    A truncated download usually fails on the zip central directory, but a
    file can also be the right size and still be garbage, so actually load it
    rather than trusting st_size.
    """
    if not path.exists():
        return False
    try:
        torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:
        return False


def _atomic_put(write_body, dest: Path) -> None:
    """Materialize `dest` via a private temp file plus os.replace.

    The temp name carries the pid so concurrent callers never share it, and
    os.replace is atomic within a filesystem -- a reader sees either the old
    file or the complete new one, never a partial write.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            write_body(f)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_inception_weights(verbose: bool = True) -> Path:
    """Guarantee cleanfid can load its weights from /tmp on this node.

    Call once before the first cleanfid call in a process. Returns the staged
    path. Safe to call concurrently from any number of tasks on any number of
    nodes.
    """
    shared = SHARED_CACHE_DIR / WEIGHT_NAME
    node_local = CLEANFID_CACHE_DIR / WEIGHT_NAME

    if not _loads_cleanly(shared):
        if verbose:
            print(f"Downloading Inception weights to {shared} ...", flush=True)

        def _download(f):
            with urllib.request.urlopen(INCEPTION_URL) as response:
                shutil.copyfileobj(response, f)

        _atomic_put(_download, shared)
        if not _loads_cleanly(shared):
            raise RuntimeError(
                f"Downloaded {shared} but it does not load as TorchScript. "
                "Delete it and retry."
            )

    if _loads_cleanly(node_local):
        if verbose:
            print(f"Inception weights already staged at {node_local}.", flush=True)
        return node_local

    if node_local.exists() and verbose:
        # The exact state that made a plain requeue unreliable.
        print(
            f"WARNING: {node_local} exists but is corrupt (likely a partial "
            "download from a co-resident task). Replacing it.",
            flush=True,
        )

    if verbose:
        print(f"Staging Inception weights to {node_local} ...", flush=True)
    def _copy(f):
        with open(shared, "rb") as src:
            shutil.copyfileobj(src, f)

    _atomic_put(_copy, node_local)

    if not _loads_cleanly(node_local):
        raise RuntimeError(f"Staged {node_local} but it still does not load.")
    return node_local
