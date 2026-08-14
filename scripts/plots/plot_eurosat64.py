# WRITTEN BY CLAUDE

"""EuroSAT64 sweep plots (6 train dists x 4 schedules, mean +/- std across
seeds). Same shape as plot_eurosat.py, but eurosat64 was trained after the
naming.py rewrite, so its aggregated-results keys use the canonical
ds-/cond-/dist- scheme (see src/naming.py) rather than the legacy
"eurosat_<dist>" prefix. See plot_aggregated_fid.py for the shared plotting
logic and plot_style.py for the color/linestyle scheme shared with
plot_mnist.py/plot_eurosat.py.
"""

from scripts.plots.plot_aggregated_fid import run

_DIST_PREFIX = "ds-eurosat64__cond-none__dist-"

EUROSAT64_DISTS = [
    f"{_DIST_PREFIX}uniform",
    f"{_DIST_PREFIX}logit_normal_mu_0.0_sigma_1.0",
    f"{_DIST_PREFIX}logit_normal_mu_0.0_sigma_0.3",
    f"{_DIST_PREFIX}logit_normal_mu_1.5_sigma_1.0",
    f"{_DIST_PREFIX}logit_normal_mu_-1.5_sigma_1.0",
    f"{_DIST_PREFIX}plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]

if __name__ == "__main__":
    run(
        dists=EUROSAT64_DISTS,
        prefix=_DIST_PREFIX,
        dataset_label="EuroSAT64",
        plots_dir="plots/eurosat64",
        combined_name="fid_combined_all_eurosat64.png",
    )
