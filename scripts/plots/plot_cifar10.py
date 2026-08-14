# WRITTEN BY CLAUDE

"""CIFAR-10 sweep plots (6 train dists x 4 schedules, mean +/- std across
seeds). See plot_aggregated_fid.py for the shared plotting logic and
plot_style.py for the color/linestyle scheme shared with plot_mnist.py /
plot_eurosat.py.
"""

from scripts.plots.plot_aggregated_fid import run

CIFAR10_PREFIX = "ds-cifar10__cond-none__dist-"
CIFAR10_DISTS = [
    f"{CIFAR10_PREFIX}uniform",
    f"{CIFAR10_PREFIX}logit_normal_mu_0.0_sigma_1.0",
    f"{CIFAR10_PREFIX}logit_normal_mu_0.0_sigma_0.3",
    f"{CIFAR10_PREFIX}logit_normal_mu_1.5_sigma_1.0",
    f"{CIFAR10_PREFIX}logit_normal_mu_-1.5_sigma_1.0",
    f"{CIFAR10_PREFIX}plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]

if __name__ == "__main__":
    run(
        dists=CIFAR10_DISTS,
        prefix=CIFAR10_PREFIX,
        dataset_label="CIFAR-10",
        plots_dir="plots/cifar10",
        combined_name="fid_combined_all_cifar10.png",
    )
