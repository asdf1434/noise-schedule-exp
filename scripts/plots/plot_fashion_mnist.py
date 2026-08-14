# WRITTEN BY CLAUDE

"""Fashion-MNIST sweep plots (6 train dists x 4 schedules, mean +/- std
across seeds). See plot_aggregated_fid.py for the shared plotting logic and
plot_style.py for the color/linestyle scheme shared with plot_mnist.py /
plot_eurosat.py.
"""

from scripts.plots.plot_aggregated_fid import run

FASHION_MNIST_PREFIX = "ds-fashion_mnist__cond-none__dist-"
FASHION_MNIST_DISTS = [
    f"{FASHION_MNIST_PREFIX}uniform",
    f"{FASHION_MNIST_PREFIX}logit_normal_mu_0.0_sigma_1.0",
    f"{FASHION_MNIST_PREFIX}logit_normal_mu_0.0_sigma_0.3",
    f"{FASHION_MNIST_PREFIX}logit_normal_mu_1.5_sigma_1.0",
    f"{FASHION_MNIST_PREFIX}logit_normal_mu_-1.5_sigma_1.0",
    f"{FASHION_MNIST_PREFIX}plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]

if __name__ == "__main__":
    run(
        dists=FASHION_MNIST_DISTS,
        prefix=FASHION_MNIST_PREFIX,
        dataset_label="Fashion-MNIST",
        plots_dir="plots/fashion_mnist",
        combined_name="fid_combined_all_fashion_mnist.png",
    )
