# WRITTEN BY CLAUDE

"""Shared color/linestyle scheme so MNIST and EuroSAT combined FID plots stay
visually consistent: color = training distribution, linestyle = sampling
schedule, keyed off the dataset-agnostic base experiment name (i.e. with any
"eurosat_" prefix stripped).
"""

import matplotlib.pyplot as plt

CANONICAL_DISTS = [
    "logit_normal_mu_-1.5_sigma_1.0",
    "logit_normal_mu_0.0_sigma_0.3",
    "logit_normal_mu_0.0_sigma_1.0",
    "logit_normal_mu_1.5_sigma_1.0",
    "plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
    "uniform",
]

_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DIST_COLORS = {dist: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, dist in enumerate(CANONICAL_DISTS)}

SCHEDULE_LINESTYLES = {
    "uniform": "-.",
    "shifted_coarse": "--",
    "shifted_fine": ":",
    "logit_normal": "-",
}

# For single-encoding plots (one color per schedule) rather than the
# combined plot's color=dist/linestyle=schedule encoding.
SCHEDULE_COLORS = {
    schedule: _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
    for i, schedule in enumerate(["uniform", "shifted_coarse", "shifted_fine", "logit_normal"])
}


def dist_color(dist_name: str) -> str:
    """Looks up a color by base dist name, stripping a leading dataset prefix
    (e.g. "eurosat_uniform" -> "uniform", or the canonical
    "ds-eurosat64__cond-none__dist-uniform" -> "uniform") so cross-dataset
    plots share colors.
    """
    canonical = dist_name
    if "__dist-" in canonical:
        canonical = canonical.rsplit("__dist-", 1)[1]
    elif canonical.startswith("eurosat_"):
        canonical = canonical[len("eurosat_"):]
    return DIST_COLORS[canonical]
