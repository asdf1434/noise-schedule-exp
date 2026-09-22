"""p10: why the Appendix B.4 closed form is zero over the range that matters.

Top row answers "you're averaging over many noise patterns and images, right?"
directly: it plots how many images the Eq. 70 average actually runs over. Where
that is 1, the posterior is a point mass, every term in the average is exactly
zero, and no amount of averaging produces a nonzero result.

Bottom row is the resulting mmse against the sigma^2 ceiling, with the noise
levels the swept sweeps found best marked, so the overlap between "collapsed"
and "where training happens" is visible rather than argued.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# best swept training centre at k=1, from the centre sweeps
TUNED = {"mnist": 1.0, "cifar10": 0.5}
C_MAIN = "#2f855a"
C_CEIL = "#718096"
C_TUNED = "#2b6cb0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="results/posterior_collapse")
    p.add_argument("--out_dir", default="plots/sept16")
    args = p.parse_args()

    names = [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), sharex="col")
    for col, (name, title) in enumerate(names):
        path = os.path.join(args.in_dir, f"{name}.json")
        if not os.path.exists(path):
            continue
        rec = json.load(open(path))
        sigma = np.array([r["sigma"] for r in rec["rows"]])
        eff = np.array([r["effective_n"] for r in rec["rows"]])
        mmse = np.array([r["mmse"] for r in rec["rows"]])
        ambiguous = np.array([r["frac_ambiguous"] for r in rec["rows"]])
        tuned = TUNED[name]

        # where the posterior is still a single image
        collapsed = sigma[ambiguous == 0]
        top = collapsed.max() if len(collapsed) else sigma[0]

        for ax in axes[:, col]:
            ax.axvspan(sigma[0], top, color="#c53030", alpha=0.08)
            ax.axvline(tuned, color=C_TUNED, lw=1.6, ls="--")
            ax.set_xscale("log")
            ax.set_xlim(sigma[0], sigma[-1])
            ax.grid(alpha=0.2, which="both")

        ax = axes[0, col]
        ax.plot(sigma, eff, color=C_MAIN, lw=2.8)
        ax.axhline(1.0, color="#444", lw=1.2, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 1e5)
        ax.set_title(
            f"{title}   (D={rec['dim']}, N={rec['n_images']}, "
            f"nearest neighbour {rec['nearest_neighbour_distance']:.1f})",
            fontsize=11.5,
        )
        ax.text(
            sigma[0] * 1.15,
            2.2,
            f"posterior is ONE image up to $\\sigma$={top:.2g}\n"
            f"0 of {rec['rows'][0]['n_draws']} draws show any spread there",
            fontsize=9,
            color="#c53030",
            va="bottom",
        )
        ax.annotate(
            f"best swept centre  $\\sigma$={tuned:g}",
            xy=(tuned, 2e4),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=C_TUNED,
            rotation=90,
            va="top",
        )

        ax = axes[1, col]
        ax.plot(sigma, np.maximum(mmse, 1e-16), color=C_MAIN, lw=2.8,
                label="B.4 closed-form mmse")
        ax.plot(sigma, sigma**2, color=C_CEIL, lw=1.7, ls=":",
                label=r"ceiling  $\sigma^2$")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e4)
        ax.set_xlabel(r"noise level  $\sigma$")

    axes[0, 0].set_ylabel("images the Eq. 70 average\nactually runs over")
    axes[1, 0].set_ylabel("per-pixel MMSE")
    axes[1, 0].legend(fontsize=9, loc="upper left")
    fig.suptitle(
        "Why the closed-form MMSE over the dataset is zero where training happens",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.012,
        "Shaded red: noise levels where the posterior is a point mass on the true image, so tr Cov(x0 | x_sigma) = 0 for EVERY draw and the Eq. 70 average is a list of exact zeros.\n"
        "More samples cannot help -- P(a second image is ever competitive) goes as exp(-d^2/8 sigma^2), which is 3e-100 at sigma=0.2. Dashed blue is the training noise level the sweep found best:\n"
        "the average there runs over 1.17 images on MNIST and 1.00 on CIFAR-10, giving mmse of 5e-4 and 1e-8 against sigma^2 ceilings of 1 and 0.25. The closed form is measuring when a\n"
        "nearest-neighbour lookup breaks down, not where denoising is hard.",
        ha="center",
        fontsize=8.4,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.085, 1, 0.945])
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "p10_posterior_collapse.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
