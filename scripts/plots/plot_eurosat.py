# WRITTEN BY CLAUDE

"""EuroSAT sweep plots (6 train dists x 4 schedules, mean +/- std across 20
seeds). See plot_aggregated_fid.py for the shared plotting logic and
plot_style.py for the color/linestyle scheme shared with plot_mnist.py.
"""

from scripts.plots.plot_aggregated_fid import run

EUROSAT_DISTS = [
    "eurosat_uniform",
    "eurosat_logit_normal_mu_0.0_sigma_1.0",
    "eurosat_logit_normal_mu_0.0_sigma_0.3",
    "eurosat_logit_normal_mu_1.5_sigma_1.0",
    "eurosat_logit_normal_mu_-1.5_sigma_1.0",
    "eurosat_plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]

if __name__ == "__main__":
    run(
        dists=EUROSAT_DISTS,
        prefix="eurosat_",
        dataset_label="EuroSAT",
        plots_dir="plots/eurosat",
        combined_name="fid_combined_all_eurosat.png",
    )
