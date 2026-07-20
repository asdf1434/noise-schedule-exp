"""Canonical experiment-name construction/parsing -- the one place that knows
the format, so train.py/evaluate_fid.py/scripts/plots/* can't drift apart.

Format: ds-{dataset}__cond-{conditioning}__dist-{train_dist}[_{k}_{v}]*__seed-{seed}

Every axis is always present -- no omitting "mnist" or "none" like the old
scheme did -- and "__" is reserved as the only delimiter *between* axes, never
used inside a dataset name, conditioning name, or dist_params key/value. That
means `name.split("__")` always yields exactly 4 tokens, regardless of how
many dist_params are set or how many datasets/conditioning types exist.

Example: ds-eurosat__cond-class__dist-logit_normal_mu_1.5_sigma_1.0__seed-2
"""

import re

_FIELD_RE = re.compile(r"^(?P<key>[a-z]+)-(?P<value>.*)$")
_REQUIRED_KEYS = ("ds", "cond", "dist", "seed")


def make_exp_name(
    dataset: str, conditioning: str, train_dist: str, dist_params: dict, seed: int
) -> str:
    param_suffix = "".join(f"_{k}_{v}" for k, v in dist_params.items())
    dist_token = f"{train_dist}{param_suffix}"
    return f"ds-{dataset}__cond-{conditioning}__dist-{dist_token}__seed-{seed}"


def parse_exp_name(name: str) -> dict:
    """Returns {"dataset", "conditioning", "train_dist_full", "seed"}.

    `train_dist_full` is train_dist + its params still concatenated (e.g.
    "logit_normal_mu_0.0_sigma_1.0") -- there's no need to split params back
    into a dict for any current consumer, so this doesn't attempt it.
    """
    parts = name.split("__")
    if len(parts) != len(_REQUIRED_KEYS):
        raise ValueError(
            f"'{name}' doesn't look like a canonical exp_name (expected "
            f"{len(_REQUIRED_KEYS)} '__'-separated fields, got {len(parts)})"
        )

    fields = {}
    for part in parts:
        m = _FIELD_RE.match(part)
        if not m:
            raise ValueError(f"Field '{part}' in '{name}' isn't '<key>-<value>'")
        fields[m.group("key")] = m.group("value")

    missing = set(_REQUIRED_KEYS) - fields.keys()
    if missing:
        raise ValueError(f"'{name}' is missing field(s): {sorted(missing)}")

    return {
        "dataset": fields["ds"],
        "conditioning": fields["cond"],
        "train_dist_full": fields["dist"],
        "seed": int(fields["seed"]),
    }


def base_name(name: str) -> str:
    """exp_name with the seed field stripped -- for grouping seeds of the same config."""
    parts = name.split("__")
    if len(parts) != len(_REQUIRED_KEYS):
        raise ValueError(f"'{name}' doesn't look like a canonical exp_name")
    return "__".join(parts[:-1])
