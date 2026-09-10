# WRITTEN BY CLAUDE

"""Where InfoNoise decided to train, across data scales.

Reads logs/metrics/<exp>/infonoise_profile.jsonl (appended once per refresh by
src/infonoise.py) and draws, on a shared log-sigma axis:

  1. rho_hat -- the estimated conditional-entropy-rate profile, which is the
     quantity the I-MMSE identity says should govern allocation, and the one
     that translates by exactly k when the data is scaled by k.
  2. pi -- the sampling density actually trained on. pi = rho_hat / w with the
     loss weight w fixed in sigma, so pi does NOT translate by the full k; the
     conversion drags its peak back toward sigma ~ 1. Read a claim about "did
     it follow the shift" off rho_hat, not pi.
  3. the noise levels each inference schedule's steps actually visit, so you
     can see whether the sampler ever goes where the information moved.

Densities are per unit log-sigma and normalized to unit mass, which is what
makes the three data scales comparable despite their different grids
(mnist_x10 raises sigma_max to 400, mnist_x0.1 lowers sigma_min to 2e-4).

    python scripts/plots/plot_infonoise_allocation.py
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Data scale -> the noise level a correctly-tracking profile should centre on.
# mmse_k(sigma) = k^2 mmse_1(sigma/k), so scaling the data by k translates the
# entropy-rate profile by exactly k along sigma.
SCALES = {
    "mnist": 1.0,
    "mnist_x10": 10.0,
    "mnist_x0.1": 0.1,
}

# The gated arm to treat as canonical per scale: the pivot rescaled by k, which
# is the arm each sweep was designed around. Selecting "the first gated arm"
# instead would pick mnist's gate_c=0.0, the deliberately ungated control whose
# whole point is that its estimate blows up at the low-noise endpoint.
CANONICAL_GATE = {
    "mnist": "0.15",
    "mnist_x10": "1.5",
    "mnist_x0.1": "0.015",
}

# The fixed logit-normal arms each sweep ran, as mu on t. logit(t) = -log(sigma),
# so a logit-normal(mu, s) on t is exactly a normal(-mu, s) on log(sigma) --
# which is why these are drawable on the same axis as the learned density with
# no reweighting. mu=0 centres sigma=1; mu=-2.3026 centres sigma=10.
LOGIT_NORMAL_ARMS = {
    "mnist": [(0.0, 1.0, "prior = tuned  logit-normal(mu=0)")],
    "mnist_x10": [
        (0.0, 1.0, "prior  logit-normal(mu=0)  -> sigma 1"),
        (-2.3026, 1.0, "tuned  logit-normal(mu=-2.30)  -> sigma 10"),
    ],
    "mnist_x0.1": [
        (0.0, 1.0, "prior  logit-normal(mu=0)  -> sigma 1"),
        (2.3026, 1.0, "tuned  logit-normal(mu=+2.30)  -> sigma 0.1"),
    ],
}


def logit_normal_on_log_sigma(mu, width, grid):
    """logit-normal(mu, width) on t, expressed as a density over log(sigma)."""
    z = (np.log(grid) + mu) / width
    dens = np.exp(-0.5 * z**2)
    return dens / dens.sum()


def _median(grid, dens):
    return grid[np.searchsorted(np.cumsum(dens) / dens.sum(), 0.5)]

# num_steps and the shift values train.py evaluates at every checkpoint.
NUM_STEPS = 50
SHIFTS = {
    "shifted_s0.15": 0.15, "shifted_coarse": 0.3, "shifted_s0.5": 0.5,
    "shifted_s0.7": 0.7, "shifted_s1.5": 1.5, "shifted_fine": 3.0,
    "shifted_s5.0": 5.0,
}


def schedule_sigmas() -> dict:
    """Interior step noise levels per schedule. sigma = (1-t)/t.

    The t=0 and t=1 endpoints are dropped: they are sigma=inf and sigma=0, carry
    no information about coverage, and t=1 comes back as ~2e-16 rather than 0 in
    floating point, which otherwise makes every schedule look like it reaches
    arbitrarily low noise.
    """
    from scipy.special import ndtri

    out = {}
    out["uniform"] = np.linspace(0.0, 1.0, NUM_STEPS + 1)
    eps = 1e-5
    p = np.linspace(eps, 1.0 - eps, NUM_STEPS + 1)
    out["logit_normal"] = np.clip(1.0 / (1.0 + np.exp(-ndtri(p))), 0.0, 1.0)
    u = np.linspace(0.0, 1.0, NUM_STEPS + 1)
    for name, s in SHIFTS.items():
        out[name] = u * s / (1.0 + (s - 1.0) * u)
    return {n: (1.0 - t[1:-1]) / t[1:-1] for n, t in out.items()}


def _arm_label(dist: str) -> str:
    gate = re.search(r"gate_c_([\d.]+)", dist)
    return f"gate_c={gate.group(1)}" if gate else "auto gate"


def load_profiles(dataset: str):
    """-> {dist: (sigma_centers, {'rho_hat': [seeds x bins], 'pi': ...}, n_seeds)}

    Uses the LAST refresh of each run. A resumed run appends a second pass over
    the same refresh numbers, so rows are deduped by refresh before taking it.
    """
    runs = defaultdict(list)
    pattern = f"logs/metrics/ds-{dataset}__cond-none__dist-infonoise*/infonoise_profile.jsonl"
    for path in sorted(glob.glob(pattern)):
        dist = path.split("dist-")[1].split("__seed")[0]
        by_refresh = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                by_refresh[row["refresh"]] = row
        if by_refresh:
            runs[dist].append(by_refresh[max(by_refresh)])

    out = {}
    for dist, rows in runs.items():
        sigma = np.array(rows[0]["sigma_centers"])
        fields = {}
        for key in ("rho_hat", "pi"):
            stack = np.array([r[key] for r in rows], dtype=float)
            # per unit log-sigma, unit mass -- the grid is log-spaced so the bin
            # width is constant and normalizing the sum suffices
            stack = stack / stack.sum(axis=1, keepdims=True)
            fields[key] = stack
        out[dist] = (sigma, fields, len(rows))
    return out


def _draw(ax, sigma, stack, color, label, ls="-"):
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else 0
    ax.plot(sigma, mean, ls, color=color, lw=2.0, label=label)
    ax.fill_between(sigma, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)


def figure_allocation(save_dir):
    """rho_hat and pi at the final refresh, all three data scales overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharex=True)
    colors = {"mnist": "#4C72B0", "mnist_x10": "#DD8452", "mnist_x0.1": "#55A868"}

    for ds, k in SCALES.items():
        profiles = load_profiles(ds)
        if not profiles:
            print(f"[warn] no profile logs for {ds}")
            continue
        # the gated arm is the headline; auto is drawn dashed alongside it
        want = CANONICAL_GATE.get(ds)
        gated = [d for d in profiles if re.search(rf"gate_c_{re.escape(want)}$", d)]
        auto = [d for d in profiles if "gate_c" not in d]
        for dists, ls in ((sorted(gated)[:1], "-"), (sorted(auto)[:1], "--")):
            for dist in dists:
                sigma, fields, n = profiles[dist]
                tag = f"{ds} (k={k:g}) - {_arm_label(dist)}, n={n}"
                _draw(axes[0], sigma, fields["rho_hat"], colors[ds], tag, ls)
                _draw(axes[1], sigma, fields["pi"], colors[ds], tag, ls)

    # where a correctly-tracking profile should sit for each scale
    for ax in axes:
        for ds, k in SCALES.items():
            ax.axvline(k, color=colors[ds], ls=":", lw=1.4, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(r"noise level  $\sigma = (1-t)/t$")
        ax.grid(alpha=0.3, which="both")

    # InfoNoise's warm-up prior, logit-normal(0,1) on t. logit(t) = -log(sigma),
    # so in log-sigma the prior is exactly a standard normal centred on sigma=1.
    grid = np.logspace(-4, 2.7, 400)
    prior = np.exp(-0.5 * np.log(grid) ** 2)
    prior /= prior.sum()
    for ax in axes:
        ax.plot(grid, prior, color="0.35", lw=1.4, ls="-.", label="warm-up prior logit-normal(0,1)")

    axes[0].set_title(r"$\hat{\rho}$ — estimated entropy-rate profile"
                      "\n(this is what should translate by $k$)", fontsize=12, fontweight="bold")
    axes[1].set_title(r"$\pi$ — sampling density actually trained on"
                      "\n($=\\hat{\\rho}/w$, so it shifts less than $k$)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("normalized density per unit $\\log\\sigma$")
    axes[1].legend(fontsize=8.5, loc="upper right", frameon=True, framealpha=0.9)
    fig.suptitle("Where InfoNoise decided to train, by data scale (final refresh, mean ± s.e.m. over seeds)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "infonoise_learned_allocation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_coverage(save_dir):
    """Learned allocation against the noise levels each sampler actually visits."""
    sched = schedule_sigmas()
    order = ["shifted_s0.15", "shifted_coarse", "shifted_s0.5", "shifted_s0.7",
             "uniform", "logit_normal", "shifted_s1.5", "shifted_fine", "shifted_s5.0"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    colors = {"mnist": "#4C72B0", "mnist_x10": "#DD8452", "mnist_x0.1": "#55A868"}

    for ax, (ds, k) in zip(axes, SCALES.items()):
        profiles = load_profiles(ds)
        want = CANONICAL_GATE.get(ds)
        gated = [d for d in profiles if re.search(rf"gate_c_{re.escape(want)}$", d)]
        if gated:
            sigma, fields, n = profiles[gated[0]]
            _draw(ax, sigma, fields["rho_hat"], colors[ds],
                  rf"$\hat{{\rho}}$  {_arm_label(gated[0])}, n={n}")
            mean = fields["rho_hat"].mean(axis=0)
            cdf = np.cumsum(mean)
            lo, hi = sigma[np.searchsorted(cdf, 0.05)], sigma[np.searchsorted(cdf, 0.95)]
            ax.axvspan(lo, hi, color=colors[ds], alpha=0.10, lw=0,
                       label=rf"central 90% of $\hat{{\rho}}$  [{lo:.3g}, {hi:.3g}]")
            top = mean.max()
        else:
            top = 1.0

        # step positions per schedule, as rows of ticks beneath the curve
        for i, name in enumerate(order):
            y = -top * (0.10 + 0.075 * i)
            sig = sched[name]
            ax.plot(sig, np.full_like(sig, y), "|", ms=7, color="0.3", mew=1.2)
            ax.text(ax.get_xlim()[0] if False else 1.2e-4, y, name,
                    fontsize=8, va="center", ha="left", color="0.25")

        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1e3)
        ax.set_ylim(-top * (0.10 + 0.075 * len(order)) - top * 0.05, top * 1.15)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(k, color=colors[ds], ls=":", lw=1.5)
        ax.set_title(f"{ds}  (data scale k={k:g}; profile should centre near sigma={k:g})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.25, which="both", axis="x")
        ax.set_yticks([])
    axes[-1].set_xlabel(r"noise level  $\sigma = (1-t)/t$")
    fig.suptitle("Learned allocation vs. what the samplers actually visit",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "infonoise_vs_sampler_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_vs_logit_normal(save_dir):
    """The like-for-like comparison: learned pi against the fixed arms it competed with.

    Uses pi, not rho_hat. pi is the distribution training samples were actually
    drawn from, which is the same kind of object as a logit-normal --el rho_hat is
    the target profile and is not directly comparable to a training distribution.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12), sharex=True)
    colors = {"mnist": "#4C72B0", "mnist_x10": "#DD8452", "mnist_x0.1": "#55A868"}
    grid = np.logspace(-4.2, 3.0, 700)

    for ax, (ds, k) in zip(axes, SCALES.items()):
        profiles = load_profiles(ds)
        want = CANONICAL_GATE.get(ds)

        for mu, width, label in LOGIT_NORMAL_ARMS[ds]:
            dens = logit_normal_on_log_sigma(mu, width, grid)
            style = dict(color="0.45", ls="--") if mu == 0.0 else dict(color="0.15", ls="-.")
            ax.plot(grid, dens, lw=2.0, label=label, **style)

        for dist in sorted(profiles, key=lambda d: "gate_c" not in d):
            is_auto = "gate_c" not in dist
            if not is_auto and not re.search(rf"gate_c_{re.escape(want)}$", dist):
                continue
            sigma, fields, n = profiles[dist]
            stack = fields["pi"]
            ls = ":" if is_auto else "-"
            _draw(ax, sigma, stack, colors[ds],
                  f"InfoNoise learned, {_arm_label(dist)}", ls)

        ymax = ax.get_ylim()[1]
        ax.axvline(k, color="0.3", ls=":", lw=1.5)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1e3)
        ax.set_ylim(0, ymax * 1.05)
        ax.set_title(f"{ds}   —   informative region near sigma = {k:g}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("density per unit $\\log\\sigma$")
        ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
        ax.grid(alpha=0.25, which="both")

    axes[-1].set_xlabel(r"noise level  $\sigma = (1-t)/t$")
    fig.suptitle("What each method chose to train on", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "infonoise_vs_logit_normal.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="plots/infonoise_scaling")
    args = parser.parse_args()
    figure_allocation(args.save_dir)
    figure_vs_logit_normal(args.save_dir)
    figure_coverage(args.save_dir)


if __name__ == "__main__":
    main()
