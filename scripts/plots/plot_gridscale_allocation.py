"""Where InfoNoise actually trains, before and after scaling its estimator grid.

Run A of the scale-shift follow-up (scripts/slurm/run_gridscale.sh). For data
multiplied by k, the entropy-rate profile translates by exactly k, so a correct
allocation should translate with it. This plots the sampled density pi against
that expectation.

pi is stored per unit log-sigma (src/infonoise.py normalizes sum(pi)*d_log = 1),
so the curves are directly comparable on a log-sigma axis and the mass per bin
is pi itself, not pi*sigma.

The logit-normal(0, 1) reference is analytic: t = sigmoid(mu + s*z) gives
log sigma = -(mu + s*z), i.e. log sigma ~ Normal(-mu, s^2). For mu=0, s=1 that
is a standard normal in log sigma, centred at sigma = 1 whatever the data does.

Run from the repo root (needs logs/metrics/<exp>/infonoise_profile.jsonl):
    python -m scripts.plots.plot_gridscale_allocation
"""

import json
import os
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "plots/infonoise_scaling/gridscale_allocation.png"

# (label, experiment-name prefix, k, colour, linestyle)
ARMS = [
    ("MNIST, default grid", "ds-mnist__cond-none__dist-infonoise", 1.0, "#4C72B0", "-"),
    ("MNIST x10, default grid", "ds-mnist_x10__cond-none__dist-infonoise_sigma_max_400.0", 10.0, "#DD8452", "--"),
    ("MNIST x10, scaled grid", "ds-mnist_x10__cond-none__dist-infonoise_sigma_min_0.02_sigma_max_800.0", 10.0, "#DD8452", "-"),
    ("MNIST x0.1, default grid", "ds-mnist_x0.1__cond-none__dist-infonoise_sigma_min_0.0002", 0.1, "#55A868", "--"),
    ("MNIST x0.1, scaled grid", "ds-mnist_x0.1__cond-none__dist-infonoise_sigma_min_0.0002_sigma_max_8.0", 0.1, "#55A868", "-"),
    ("CIFAR-10 x10, scaled grid", "ds-cifar10_x10__cond-none__dist-infonoise_sigma_min_0.02_sigma_max_800.0", 10.0, "#C44E52", "-"),
    ("CIFAR-10 x0.1, scaled grid", "ds-cifar10_x0.1__cond-none__dist-infonoise_sigma_min_0.0002_sigma_max_8.0", 0.1, "#8172B3", "-"),
]

# Panels are grouped by shift direction so each group shares an x range.
PANELS = [(0.1, "k = 0.1"), (1.0, "k = 1 (control)"), (10.0, "k = 10")]


def upper_mode(centers, dens):
    """Sigma of the highest-sigma local maximum of the density.

    The x0.1 arms are bimodal: a spike pinned at the grid's low edge plus the
    lobe that reflects the data. A plain argmax reports the spike, which is a
    grid artifact rather than where training mass meaningfully sits. Taking the
    largest-sigma peak returns the same value on the unimodal arms and the
    substantive lobe on the bimodal ones.
    """
    interior = np.where((dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:]))[0] + 1
    if len(interior) == 0:
        return float(centers[int(np.argmax(dens))])
    return float(centers[interior[-1]])


def gmean(a):
    return float(np.exp(np.mean(np.log(np.asarray(a, dtype=float)))))


def load_arm(prefix):
    """Final-refresh pi per seed, on the shared sigma grid of the first seed."""
    centers, curves, gates, modes = None, [], [], []
    for d in sorted(glob.glob(f"logs/metrics/{prefix}__seed-*")):
        path = os.path.join(d, "infonoise_profile.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        if not lines:
            continue
        last = json.loads(lines[-1])
        c = np.asarray(last["sigma_centers"], dtype=float)
        pi = np.asarray(last["pi"], dtype=float)
        if centers is None:
            centers = c
        elif c.shape != centers.shape or not np.allclose(c, centers):
            # Arms with different grids never share a curve array; skip the
            # mismatch rather than silently interpolating onto the wrong axis.
            continue
        curves.append(pi)
        gates.append(float(last["gate_c"]))
        modes.append(upper_mode(c, pi))
    if not curves:
        return None
    return centers, np.stack(curves), gates, modes


def main():
    loaded = {}
    for label, prefix, k, color, ls in ARMS:
        got = load_arm(prefix)
        if got is None:
            print(f"  (no profile logs for {label} -- skipping)")
            continue
        loaded[label] = (got, k, color, ls)

    if not loaded:
        raise SystemExit("no infonoise profile logs found under logs/metrics/")

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)

    # --- top row: the distributions, one panel per data scale ---------------
    for col, (k_panel, title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, col])
        for label, ((centers, curves, _g, _m), k, color, ls) in loaded.items():
            if k != k_panel:
                continue
            mean = curves.mean(axis=0)
            sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else None
            ax.plot(centers, mean, color=color, ls=ls, lw=1.8, label=label)
            if sem is not None:
                ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)

        # logit-normal(0, 1) prior: standard normal in log sigma, always at 1
        grid = np.logspace(-4, 3, 400)
        prior = np.exp(-0.5 * np.log(grid) ** 2) / np.sqrt(2 * np.pi)
        ax.plot(grid, prior, color="0.45", ls=":", lw=1.6, label="logit-normal(0, 1) prior")

        # where a correctly-translating allocation would sit: MNIST's k=1 mode
        # (2.05) carried over by k.
        ax.axvline(2.052 * k_panel, color="k", ls="-.", lw=1.2, alpha=0.7,
                   label=f"expected centre ({2.052 * k_panel:.3g})")

        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1e3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"noise level  $\sigma = (1-t)/t$")
        if col == 0:
            ax.set_ylabel(r"density per unit $\log\sigma$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")

    # --- bottom left: mode of pi vs k, against the ideal translation --------
    ax = fig.add_subplot(gs[1, 0])
    ks = np.array([0.1, 1.0, 10.0])
    ax.plot(ks, 2.052 * ks, color="k", ls="-.", lw=1.4, label="correct translation (slope 1)")
    for label, ((centers, curves, _g, modes), k, color, ls) in loaded.items():
        marker = "o" if ls == "-" else "x"
        ax.plot([k], [gmean(modes)], marker=marker, ms=9, color=color,
                ls="none", label=label, mfc="none" if marker == "o" else color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("data scale k")
    ax.set_ylabel(r"highest-$\sigma$ peak of $\pi$")
    ax.set_title("Where training mass ends up\n(x0.1 arms are bimodal; this is the upper lobe)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    # --- bottom middle: gate cutoff vs k -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ks, 0.0205 * ks, color="k", ls="-.", lw=1.4, label="correct translation (slope 1)")
    for label, ((centers, curves, gates, _m), k, color, ls) in loaded.items():
        marker = "o" if ls == "-" else "x"
        ax.plot([k], [gmean(gates)], marker=marker, ms=9, color=color,
                ls="none", label=label, mfc="none" if marker == "o" else color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("data scale k")
    ax.set_ylabel(r"gate cutoff $c$")
    ax.set_title("Where the gate cutoff ends up", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    # --- bottom right: the loss weight, i.e. the thing doing the pinning ----
    ax = fig.add_subplot(gs[1, 2])
    grid = np.logspace(-4, 3, 800)
    w = (1.0 + grid) ** 2 / np.maximum(0.05 * (1.0 + grid), grid) ** 2
    ax.plot(grid, w, color="#937860", lw=1.8, label=r"$w(\sigma)=1/\max(0.05,\,1-t)^2$")
    ax.axvline(1.0, color="k", ls=":", lw=1.2, label=r"fixed scale at $\sigma=1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$w$")
    ax.set_title("The loss weight has a scale in it", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Scaling InfoNoise's estimator grid fixes the gate cutoff but not the training distribution\n"
        "(final refresh, mean +/- s.e.m. over 10 seeds per arm)",
        fontsize=12,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
