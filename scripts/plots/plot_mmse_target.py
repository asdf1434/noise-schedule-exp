"""InfoNoise's estimate against the ground-truth target it is estimating.

Produces two figures, one panel per (dataset x scale) cell:

  p5_learned_distributions.png -- training densities. The centre the sweep found
      best, what InfoNoise actually learned, and what the Eq. 13-15 conversion
      produces when the closed-form mmse is substituted for the model's m_hat.
      That third curve is the schedule InfoNoise would pick if its estimator
      were perfect, so the gap to it separates a bad estimate from a bad target.

  p6_mmse_vs_mhat.png -- the profile estimates themselves, log-log. The closed
      form, the model's m_hat, and the hard ceiling mmse(sigma) <= sigma^2 that
      the trivial estimator x_hat = y attains. The density plot hides the size
      of the estimator error, because the conversion renormalizes; this does
      not.

Inputs:
  results/closed_form_mmse/{mnist,cifar10}.json  (scripts/analysis/closed_form_mmse.py)
  a profiles JSON                                 (scripts/analysis/extract_infonoise_profiles.py)

Scaled cells reuse the base curve via mmse_k(sigma) = k^2 mmse_1(sigma/k), which
is exact -- see the module docstring of closed_form_mmse.py.
"""

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.infonoise import InfoNoiseSampler  # noqa: E402

# cell -> (base dataset, scale k, panel title)
CELLS = [
    ("mnist_x0.1", "mnist", 0.1, "MNIST x0.1   (k = 0.1)"),
    ("mnist", "mnist", 1.0, "MNIST   (k = 1)"),
    ("mnist_x10", "mnist", 10.0, "MNIST x10   (k = 10)"),
    ("cifar10_x0.1", "cifar10", 0.1, "CIFAR-10 x0.1   (k = 0.1)"),
    ("cifar10", "cifar10", 1.0, "CIFAR-10   (k = 1)"),
    ("cifar10_x10", "cifar10", 10.0, "CIFAR-10 x10   (k = 10)"),
]

C_TUNED = "#2b6cb0"
C_INFO = "#c53030"
C_EMP = "#2f855a"
C_HOLD = "#805ad5"
C_GAUSS = "#b7791f"
C_CEIL = "#718096"

_FLOOR = 1e-30  # stands in for an exact zero so the interpolation can stay in log space


def rescaled_mmse(base: dict, k: float, variant: str, sigma: np.ndarray) -> np.ndarray:
    """mmse_k(sigma) = k^2 mmse_1(sigma/k), interpolated onto ``sigma``.

    Interpolation is done on log(mmse) against log(sigma): the curve spans
    ~20 orders of magnitude, and interpolating it linearly would flatten the
    whole low-noise end into the first nonzero sample.
    """
    base_sigma = np.asarray(base["sigma"])
    base_mmse = np.maximum(np.asarray(base[f"mmse_{variant}"]), _FLOOR)
    log_vals = np.interp(
        np.log(np.asarray(sigma) / k), np.log(base_sigma), np.log(base_mmse)
    )
    return k**2 * np.exp(log_vals)


def logit_normal_density(sigma, centre, width=1.0):
    """Density per unit log sigma of t = sigmoid(mu + z), z ~ N(0, width^2).

    log sigma is then normal about log(centre), which is why a logit-normal and
    InfoNoise's tabulated pi are directly comparable on these axes.
    """
    x = np.log(sigma)
    return np.exp(-0.5 * ((x - np.log(centre)) / width) ** 2) / (
        width * math.sqrt(2 * math.pi)
    )


def sampler_for(cell: dict, num_bins: int) -> InfoNoiseSampler:
    """A sampler carrying this cell's grid and loss weight, used only for its
    Eq. 13-15 conversion -- the same code path training uses."""
    return InfoNoiseSampler(
        warmup_sample_fn=lambda key, n: None,
        num_bins=num_bins,
        sigma_min=cell["sigma_min"],
        sigma_max=cell["sigma_max"],
        t_clip=cell["t_clip"],
    )


def build(profiles: dict, mmse: dict) -> dict:
    """Per cell: the two MMSE curves on the cell's own grid, and the densities
    the conversion produces from each."""
    built = {}
    for cell_name, base_name, k, title in CELLS:
        if cell_name not in profiles or base_name not in mmse:
            continue
        cell = profiles[cell_name]
        sigma = np.asarray(cell["sigma"])
        sampler = sampler_for(cell, len(sigma))
        # the sampler builds its own grid from (sigma_min, sigma_max, num_bins);
        # it must land on the one the run actually logged, or w(sigma) is wrong
        assert np.allclose(sampler.centers, sigma, rtol=1e-6), cell_name

        entry = {"title": title, "sigma": sigma, "k": k}
        for variant in ("empirical", "holdout", "gaussian"):
            curve = rescaled_mmse(mmse[base_name], k, variant, sigma)
            entry[f"mmse_{variant}"] = curve
            allocation = sampler.allocation_from_m_hat(curve)
            entry[f"pi_{variant}"] = allocation["pi"]
            entry[f"gate_c_{variant}"] = allocation["gate_c"]
        built[cell_name] = entry
    return built


def median_sigma(sigma: np.ndarray, pi: np.ndarray) -> float:
    d_log = float(np.log(sigma[1]) - np.log(sigma[0]))
    cdf = np.cumsum(pi) * d_log
    return float(np.interp(0.5, cdf / cdf[-1], sigma))


def plot_densities(profiles, built, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    for ax, (cell_name, _, _, _) in zip(axes.ravel(), CELLS):
        if cell_name not in built:
            ax.set_axis_off()
            continue
        cell, b = profiles[cell_name], built[cell_name]
        sigma, tuned = b["sigma"], cell["tuned_sigma"]
        grid = np.logspace(np.log10(tuned / 400), np.log10(tuned * 400), 900)

        ax.plot(
            grid,
            logit_normal_density(grid, tuned),
            color=C_TUNED,
            lw=2.6,
            label=f"best swept centre  ($\\sigma$={tuned:g})",
        )
        ax.fill_between(
            grid, 0, logit_normal_density(grid, tuned), color=C_TUNED, alpha=0.13
        )
        ax.plot(
            sigma,
            np.asarray(cell["pi"]),
            color=C_INFO,
            lw=2.7,
            label="InfoNoise (learned $\\hat{m}$)",
        )
        ax.plot(
            sigma,
            b["pi_gaussian"],
            color=C_GAUSS,
            lw=2.2,
            ls="-.",
            label="MMSE, Gaussian fit  (the target)",
        )

        ax.axvline(tuned, color=C_TUNED, lw=1.1, ls=":")
        med_info = cell["median"]
        med_gauss = median_sigma(sigma, b["pi_gaussian"])
        ax.text(
            0.02,
            0.97,
            f"median $\\sigma$   swept {tuned:g}\n"
            f"  InfoNoise {med_info:.3g}  ({med_info / tuned:.2f}x)\n"
            f"  MMSE Gaussian {med_gauss:.3g}  ({med_gauss / tuned:.2f}x)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=.38", fc="white", ec="#ccc", alpha=0.93),
        )
        ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, 0.62)
        ax.set_title(b["title"], fontsize=11.5)
        ax.grid(alpha=0.2, which="both")
        ax.set_xlabel(r"noise level  $\sigma$")
    for r in (0, 1):
        axes[r, 0].set_ylabel(r"training density (per unit $\log\sigma$)")
    axes[0, 0].legend(fontsize=8.2, loc="center left")
    fig.suptitle(
        "Where InfoNoise trains, against the swept optimum and against its own target "
        "computed in closed form",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.008,
        "The gold curve is the same Eq. 13-15 conversion InfoNoise runs, fed the closed-form mmse(sigma) instead of the model's estimate -- the schedule InfoNoise would pick if it\n"
        "measured the profile perfectly. Gap from red to gold is estimator error; gap from gold to blue is error in the target itself. See p7 for why the Gaussian fit is the\n"
        "closed form used here, and p6 for what goes wrong with the measurement.",
        ha="center",
        fontsize=8.4,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.075, 1, 0.945])
    path = os.path.join(out_dir, "p5_learned_distributions.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def plot_profiles(profiles, built, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    for ax, (cell_name, _, _, _) in zip(axes.ravel(), CELLS):
        if cell_name not in built:
            ax.set_axis_off()
            continue
        cell, b = profiles[cell_name], built[cell_name]
        sigma = b["sigma"]
        m_hat = np.asarray(cell["m_hat"])
        ceiling = sigma**2

        ax.plot(sigma, ceiling, color=C_CEIL, lw=1.8, ls=":", label=r"ceiling  $\sigma^2$")
        ax.plot(
            sigma,
            b["mmse_gaussian"],
            color=C_GAUSS,
            lw=2.5,
            label="closed-form MMSE  (the target)",
        )
        ax.plot(
            sigma, m_hat, color=C_INFO, lw=2.7, label=r"model $\hat{m}$ (InfoNoise)"
        )

        ax.axvline(cell["gate_c"], color=C_INFO, lw=1.2, ls=":")
        ax.axvline(b["gate_c_gaussian"], color=C_GAUSS, lw=1.2, ls=":")
        ax.annotate(
            f"$c$={cell['gate_c']:.2g}",
            xy=(cell["gate_c"], 1e-9),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=C_INFO,
            rotation=90,
            va="bottom",
            ha="left",
        )
        ax.annotate(
            f"target $c$={b['gate_c_gaussian']:.2g}",
            xy=(b["gate_c_gaussian"], 1e-9),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=C_GAUSS,
            rotation=90,
            va="bottom",
            ha="left",
        )

        over = m_hat / np.maximum(ceiling, 1e-300)
        worst = int(np.argmax(over))
        # local log-log slope over the lowest decade of the grid. Every valid
        # answer has slope 2 there (mmse ~ sigma^2 for any continuous p(x));
        # m_hat goes flat instead, and it is that shape difference -- not the
        # size of the error -- that moves the gate, since the conversion is
        # invariant to scaling m_hat by a constant.
        low = sigma <= sigma[0] * 10
        log_sigma = np.log(sigma[low])
        slope_m = float(
            np.polyfit(log_sigma, np.log(np.maximum(m_hat[low], 1e-300)), 1)[0]
        )
        slope_t = float(
            np.polyfit(log_sigma, np.log(b["mmse_gaussian"][low]), 1)[0]
        )
        ax.text(
            0.02,
            0.97,
            f"slope over the lowest decade\n"
            f"   target {slope_t:.2f}    $\\hat{{m}}$ {slope_m:.2f}\n"
            f"$\\hat{{m}}$ is up to {over[worst]:.0f}x above the $\\sigma^2$ ceiling\n"
            f"gate $c$ {b['gate_c_gaussian'] / cell['gate_c']:.0f}x below where the target puts it",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=.38", fc="white", ec="#ccc", alpha=0.93),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(sigma[0], sigma[-1])
        ax.set_ylim(1e-10, max(m_hat.max(), ceiling.max()) * 5)
        ax.set_title(b["title"], fontsize=11.5)
        ax.grid(alpha=0.2, which="both")
        ax.set_xlabel(r"noise level  $\sigma$")
    for r in (0, 1):
        axes[r, 0].set_ylabel(r"per-pixel denoising error  $\hat{m}(\sigma)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=9.5,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.115),
        frameon=False,
    )
    fig.suptitle(
        r"What InfoNoise estimates ($\hat{m}$) against what it is estimating (closed-form MMSE)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.008,
        "$\\hat{m}$ is the model's own binned unweighted denoising loss, and is the only input to the schedule. It tracks the target down to about $\\sigma = 10^{-2}k$, then flattens instead of\n"
        "continuing to fall as $\\sigma^2$, ending above the ceiling mmse$(\\sigma) \\leq \\sigma^2$ that copying the input already attains. The schedule is exactly invariant to scaling $\\hat{m}$ by a constant,\n"
        "so only that change of SHAPE matters: a flat $\\hat{m}$ divided by $\\sigma^3$ blows up at the grid floor, the gate rule reads it as an onset of information, and $c$ (dotted red) lands well\n"
        "below where the target puts it (dotted gold).",
        ha="center",
        fontsize=8.4,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.175, 1, 0.945])
    path = os.path.join(out_dir, "p6_mmse_vs_mhat.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def plot_mmse_definitions(mmse, out_dir):
    """p7: the three closed forms against each other, for the two base datasets.

    Separate from p6 because it answers a different question -- not "is the
    model's estimate right" but "what does mmse(sigma) even mean for a finite
    dataset, and which of these can serve as a target". Only the base datasets
    appear: mmse_k(sigma) = k^2 mmse_1(sigma/k) makes every scaled cell the same
    curve shifted along both axes, so plotting six panels would repeat two.
    """
    names = [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, (name, title) in zip(axes, names):
        if name not in mmse:
            ax.set_axis_off()
            continue
        rec = mmse[name]
        sigma = np.asarray(rec["sigma"])
        emp = np.asarray(rec["mmse_empirical"])
        hold = np.asarray(rec["mmse_holdout"])
        gauss = np.asarray(rec["mmse_gaussian"])

        ax.plot(sigma, sigma**2, color=C_CEIL, lw=1.8, ls=":", label=r"ceiling  $\sigma^2$")
        ax.plot(
            sigma,
            np.maximum(emp, _FLOOR),
            color=C_EMP,
            lw=2.4,
            ls="--",
            label=r"$p(x)$ = the $N$ images as point masses",
        )
        ax.plot(
            sigma,
            hold,
            color=C_HOLD,
            lw=2.4,
            ls="-.",
            label=r"same, but scoring held-out images",
        )
        ax.plot(
            sigma, gauss, color=C_GAUSS, lw=2.8, label=r"$p(x)$ = Gaussian fit  (the one used)"
        )

        # where the point-mass posterior stops collapsing onto the true image
        rising = np.nonzero(emp > 1e-6)[0]
        if len(rising):
            ax.axvline(sigma[rising[0]], color=C_EMP, lw=1.1, ls=":")
            ax.annotate(
                f"lookup stops being\nperfect: $\\sigma$={sigma[rising[0]]:.2g}",
                xy=(sigma[rising[0]], 3e-9),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8.5,
                color=C_EMP,
                va="bottom",
            )

        ax.text(
            0.02,
            0.97,
            f"held-out floor {hold[0]:.3f} per pixel\n"
            f"  = mean sq. distance to the nearest\n"
            f"    training image ({rec['n_support']} of them)\n"
            f"all three agree above $\\sigma \\approx$ 10, at the pixel variance",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.6,
            bbox=dict(boxstyle="round,pad=.38", fc="white", ec="#ccc", alpha=0.93),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(sigma[0], sigma[-1])
        ax.set_ylim(1e-9, 1e5)
        ax.set_title(f"{title}   (k = 1;  every other scale is this curve shifted)", fontsize=11.5)
        ax.grid(alpha=0.2, which="both")
        ax.set_xlabel(r"noise level  $\sigma$")
    axes[0].set_ylabel(r"per-pixel MMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=9.5,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.155),
        frameon=False,
    )
    fig.suptitle(
        "mmse$(\\sigma)$ computed in closed form, under three choices of what the data distribution is",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.012,
        "All three are exact, not approximations -- they differ only in the assumed $p(x)$. Treating the dataset as $N$ isolated points makes optimal denoising a nearest-neighbour lookup, and a lookup is\n"
        "perfect until the noise can confuse two different images, so that MMSE is exactly ZERO over most of the range and cannot serve as a target. Scoring held-out images instead removes the collapse but\n"
        "saturates at the nearest-neighbour distance, which sits above the $\\sigma^2$ ceiling and is therefore a very loose bound. The Gaussian fit is continuous, so it behaves as $\\sigma^2$ at low noise like\n"
        "real data does; it overestimates the true MMSE, but the schedule is invariant to scaling, so only its shape is used.",
        ha="center",
        fontsize=8.4,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.215, 1, 0.93])
    path = os.path.join(out_dir, "p7_mmse_definitions.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", default="results/infonoise_profiles.json")
    p.add_argument("--mmse_dir", default="results/closed_form_mmse")
    p.add_argument("--out_dir", default="plots/sept16")
    args = p.parse_args()

    profiles = json.load(open(args.profiles))
    # a base dataset whose curve has not been computed yet just drops its cells
    # from the figures, rather than failing the whole run
    mmse = {}
    for name in sorted({base for _, base, _, _ in CELLS}):
        path = os.path.join(args.mmse_dir, f"{name}.json")
        if os.path.exists(path):
            mmse[name] = json.load(open(path))
        else:
            print(f"  (no closed-form MMSE for {name}: {path} missing)")
    os.makedirs(args.out_dir, exist_ok=True)

    built = build(profiles, mmse)
    plot_densities(profiles, built, args.out_dir)
    plot_profiles(profiles, built, args.out_dir)
    plot_mmse_definitions(mmse, args.out_dir)


if __name__ == "__main__":
    main()
