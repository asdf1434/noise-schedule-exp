# WRITTEN BY CLAUDE

"""MNIST exp1/exp2 sweep plots (6 train dists x 4 schedules, mean +/- std
across seeds -- 20 for exp2's four logit_normal variants + plateau, 5 for
exp1's uniform). See plot_aggregated_fid.py for the shared plotting logic
and plot_style.py for the color/linestyle scheme shared with plot_eurosat.py.
"""

from scripts.plots.plot_aggregated_fid import run

MNIST_DISTS = [
    "uniform",
    "logit_normal_mu_0.0_sigma_1.0",
    "logit_normal_mu_0.0_sigma_0.3",
    "logit_normal_mu_1.5_sigma_1.0",
    "logit_normal_mu_-1.5_sigma_1.0",
    "plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]

if __name__ == "__main__":
    run(
        dists=MNIST_DISTS,
        prefix="",
        dataset_label="MNIST",
        plots_dir="plots/mnist",
        combined_name="fid_combined_all.png",
    )
