"""Plots for the InfoNoise generalization grid (2026-08-31 overnight run).

Four figures, written to plots/:

  infonoise_forest.png    paired InfoNoise - baseline FID deltas, 95% CI
  infonoise_ranks.png     InfoNoise's rank among the fixed training dists
  infonoise_curves.png    FID vs epoch, InfoNoise vs its logit_normal(0,1) prior
  infonoise_profiles.png  what the estimator learned, vs where each sampler looks
  infonoise_prior_distance.png  how far the learned pi moved from its warm-up prior

The last one is the diagnostic: a null in the FID table means nothing if the
profile never moved off the warm-up prior or collapsed onto an endpoint, so
check it before reading anything into the first three.

Usage:
    python -m scripts.plots.plot_infonoise_grid
    python -m scripts.plots.plot_infonoise_grid --results results/master_fid_results.cluster.json
"""

import argparse
import glob
import json
import os
import statistics as st
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.schedules import get_shifted_steps, get_uniform_steps

EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LATE = [60, 70, 80, 90, 100]  # never score by best epoch -- it rewards instability
SCHEDULES = ["uniform", "logit_normal", "shifted_coarse", "shifted_fine"]
# must match export_evaluation_images() in train.py
SHIFTS = {"shifted_coarse": 0.3, "shifted_fine": 3.0}
INFO = "infonoise_gate_c_0.15"
BASE = "logit_normal_mu_0.0_sigma_1.0"
CELLS = [
    ("ds-mnist", "cond-inpaint"),
    ("ds-fashion_mnist", "cond-none"),
    ("ds-fashion_mnist", "cond-inpaint"),
    ("ds-cifar10", "cond-none"),
    ("ds-cifar10", "cond-inpaint"),
]
MAX_SEEDS = 20


def late_mean(entry: dict) -> Optional[float]:
    """Mean FID over the last five checkpoints, or None if too few were scored."""
    v = [entry[f"epoch_{e}"] for e in LATE if f"epoch_{e}" in entry]
    return st.mean(v) if len(v) >= 4 else None


def cell_scores(results, ds, cond, dist, sched):
    """Per-seed late-epoch FID for one (cell, training dist, sampling schedule)."""
    out = {}
    for s in range(MAX_SEEDS):
        e = results.get(f"{ds}__{cond}__dist-{dist}__seed-{s}", {}).get(sched)
        if e and (m := late_mean(e)) is not None:
            out[s] = m
    return out


def fixed_dists(results, ds, cond):
    """Every training distribution present for this cell, InfoNoise excluded."""
    pref = f"{ds}__{cond}__dist-"
    names = set()
    for k in results:
        if k.startswith(pref) and "__seed-" in k:
            names.add(k[len(pref) :].rsplit("__seed-", 1)[0])
    return sorted(n for n in names if "infonoise" not in n)


# ------------------------------------------------------------------ figure 1


def plot_forest(results, out_path):
    rows = []
    for ds, cond in CELLS:
        for sched in SCHEDULES:
            a = cell_scores(results, ds, cond, INFO, sched)
            b = cell_scores(results, ds, cond, BASE, sched)
            paired = [(a[s], b[s]) for s in sorted(set(a) & set(b))]
            if len(paired) < 3:
                continue
            diffs = [x - y for x, y in paired]
            n = len(diffs)
            sem = st.stdev(diffs) / np.sqrt(n) if n > 1 else 0.0
            rows.append(
                {
                    "label": f"{ds.replace('ds-','')}/{cond.replace('cond-','')}  {sched}",
                    "mean": st.mean(diffs),
                    "ci": 1.96 * sem,
                    "n": n,
                }
            )

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        better = r["mean"] < 0
        sig = abs(r["mean"]) > r["ci"] and r["ci"] > 0
        color = ("#2166ac" if better else "#b2182b") if sig else "#999999"
        ax.errorbar(
            r["mean"], yi, xerr=r["ci"], fmt="o", color=color, capsize=3, markersize=5
        )
        ax.text(
            ax.get_xlim()[1], yi, f"  n={r['n']}", va="center", fontsize=7, color="#666"
        )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel("InfoNoise FID  -  logit_normal(0,1) FID   (negative = InfoNoise better)")
    ax.set_title(
        "Paired per-seed difference, mean of epochs 60-100\n"
        "grey = 95% CI spans zero", fontsize=10
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(rows)} comparisons)")


# ------------------------------------------------------------------ figure 2


def plot_ranks(results, out_path):
    grid, labels = [], []
    for ds, cond in CELLS:
        dists = fixed_dists(results, ds, cond)
        if not dists:
            continue
        row = []
        for sched in SCHEDULES:
            means = []
            for dist in dists + [INFO]:
                v = list(cell_scores(results, ds, cond, dist, sched).values())
                means.append((st.mean(v), dist) if len(v) >= 3 else (np.inf, dist))
            means.sort()
            order = [d for _, d in means]
            row.append(order.index(INFO) + 1 if INFO in order else np.nan)
        grid.append(row)
        labels.append(f"{ds.replace('ds-','')}/{cond.replace('cond-','')}")
    grid = np.array(grid, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(labels) + 2))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=1, vmax=7, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=11)
    ax.set_xticks(range(len(SCHEDULES)))
    ax.set_xticklabels(SCHEDULES, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("InfoNoise's rank among 7 training distributions\n(1 = best, 7 = worst)", fontsize=10)
    fig.colorbar(im, ax=ax, label="rank")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------ figure 3


def plot_curves(results, out_path):
    fig, axes = plt.subplots(1, len(CELLS), figsize=(3.1 * len(CELLS), 3.4), sharex=True)
    for ax, (ds, cond) in zip(np.atleast_1d(axes), CELLS):
        for dist, color, name in ((BASE, "#666666", "logit_normal(0,1)"), (INFO, "#1f77b4", "InfoNoise")):
            mu, se = [], []
            for e in EPOCHS:
                vals = []
                for s in range(MAX_SEEDS):
                    v = results.get(f"{ds}__{cond}__dist-{dist}__seed-{s}", {}).get(
                        "shifted_coarse", {}
                    ).get(f"epoch_{e}")
                    if v is not None:
                        vals.append(v)
                mu.append(st.mean(vals) if vals else np.nan)
                se.append(st.stdev(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            mu, se = np.array(mu), np.array(se)
            ax.plot(EPOCHS, mu, color=color, label=name, lw=1.6)
            ax.fill_between(EPOCHS, mu - se, mu + se, color=color, alpha=0.2)
        ax.set_title(f"{ds.replace('ds-','')}\n{cond.replace('cond-','')}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("FID (shifted_coarse sampling)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("FID vs training epoch, mean +/- SEM across seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------ figure 4


def step_density(sched, centers, num_steps=4000, bw=0.18):
    """Where a sampler places its model evaluations, as a density in log sigma.

    Estimated with many more steps than the 50 used at eval time: the shape of a
    schedule's step distribution does not depend on the step count, and 50 steps
    binned onto the profile's 256-bin grid is mostly empty bins. Smoothed with a
    Gaussian kernel in log sigma and normalized to unit area, so it is directly
    comparable to the learned pi plotted beside it.
    """
    ts = (
        get_uniform_steps(num_steps=num_steps)
        if sched == "uniform"
        else get_shifted_steps(num_steps=num_steps, shift=SHIFTS[sched])
    )
    ts = np.asarray(ts)
    t = np.clip(0.5 * (ts[:-1] + ts[1:]), 1e-6, 1 - 1e-6)
    sig = (1 - t) / t  # this repo's convention: z = t*x + (1-t)*noise
    ls, lc = np.log(sig), np.log(centers)
    dens = np.exp(-0.5 * ((lc[:, None] - ls[None, :]) / bw) ** 2).sum(axis=1)
    d_log = float(np.diff(lc).mean())
    return dens / max((dens * d_log).sum(), 1e-12)


def last_run(path):
    """Records of the final training run in a profile log.

    A log can hold more than one run appended end to end (the grid re-ran some
    mnist cells that the gate sweep had already produced), so take everything
    from the last refresh==1 onward.
    """
    recs = [json.loads(line) for line in open(path)]
    starts = [i for i, r in enumerate(recs) if r.get("refresh") == 1]
    return recs[starts[-1] :] if starts else recs


def plot_profiles(out_path, metrics_dir="logs/metrics"):
    cells = [(f"{ds}__{cond}__dist-{INFO}", f"{ds.replace('ds-','')}/{cond.replace('cond-','')}")
             for ds, cond in CELLS]
    cells.append((f"ds-mnist__cond-none__dist-{INFO}", "mnist/none"))

    fig, axes = plt.subplots(2, len(cells), figsize=(3.0 * len(cells), 6.0))
    for col, (exp, label) in enumerate(cells):
        paths = sorted(glob.glob(os.path.join(metrics_dir, exp + "__seed-*", "infonoise_profile.jsonl")))
        ax_p, ax_c = axes[0, col], axes[1, col]
        if not paths:
            ax_p.set_title(f"{label}\n(no profile logs)", fontsize=8)
            continue

        finals, cdf_moves = [], []
        centers = None
        for p in paths:
            recs = last_run(p)
            centers = np.array(recs[-1]["sigma_centers"])
            finals.append(np.array(recs[-1]["pi"]))
            pis = [np.array(r["pi"]) for r in recs]
            d_log = np.diff(np.log(centers)).mean()
            cdfs = [np.cumsum(pi) * d_log for pi in pis]
            cdf_moves.append([np.abs(cdfs[i] - cdfs[i - 1]).max() for i in range(1, len(cdfs))])

        mean_pi = np.mean(finals, axis=0)
        ax_p.fill_between(centers, 0, mean_pi, color="#1f77b4", alpha=0.25)
        ax_p.plot(centers, mean_pi, color="#1f77b4", lw=2.0, label="learned pi")
        for sched, style, col in (
            ("uniform", ":", "#555555"),
            ("shifted_coarse", "--", "#d95f02"),
            ("shifted_fine", "-.", "#1b9e77"),
        ):
            ax_p.plot(centers, step_density(sched, centers), style, lw=1.3,
                      color=col, label=sched)
        ax_p.set_xscale("log")
        ax_p.set_xlim(1e-2, 1e2)
        ax_p.set_title(f"{label}\n({len(paths)} seeds)", fontsize=8)
        ax_p.set_xlabel("sigma")
        ax_p.grid(alpha=0.3)
        if col == 0:
            ax_p.set_ylabel("density per unit log sigma")
            ax_p.legend(fontsize=6)

        for m in cdf_moves:
            ax_c.semilogy(range(1, len(m) + 1), m, color="#1f77b4", alpha=0.45, lw=1)
        ax_c.axhline(1e-2, color="red", ls="--", lw=1)
        ax_c.set_xlabel("refresh")
        ax_c.grid(alpha=0.3)
        if col == 0:
            ax_c.set_ylabel("max CDF change\n(converged below red)")
    fig.suptitle(
        "Top: what InfoNoise learned, against where each sampler evaluates the model.  "
        "Bottom: convergence of the estimate.", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------ figure 5


def warmup_prior_density(centers, n=400_000, seed=0):
    """The warm-up prior logit_normal(0,1), as a density in log sigma.

    Sampled rather than derived so it goes through the same coordinate change
    (sigma = (1-t)/t) the sampler itself uses.
    """
    import jax

    from src.schedules import sample_t_logit_normal

    t = np.asarray(sample_t_logit_normal(jax.random.PRNGKey(seed), n, 0.0, 1.0)).reshape(-1)
    t = np.clip(t, 1e-9, 1 - 1e-9)
    lc = np.log(centers)
    d_log = float(np.diff(lc).mean())
    edges = np.concatenate([[lc[0] - d_log / 2], (lc[:-1] + lc[1:]) / 2, [lc[-1] + d_log / 2]])
    dens, _ = np.histogram(np.log((1 - t) / t), bins=edges)
    return dens / max((dens * d_log).sum(), 1e-12)


def total_variation(a, b, d_log):
    """TV distance between two densities on a shared log-sigma grid."""
    return 0.5 * float(np.abs(a - b).sum()) * d_log


def plot_prior_distance(out_path, metrics_dir="logs/metrics"):
    """How far the learned pi ended up from the prior it warmed up on.

    This is the single number that explains the FID null: InfoNoise warms up
    from logit_normal(0,1), which is also the baseline it is compared against,
    so a small distance here means the two arms are training on nearly the same
    distribution and cannot differ much in FID.
    """
    cells = [(f"{ds}__{cond}__dist-{INFO}", f"{ds.replace('ds-','')}/{cond.replace('cond-','')}")
             for ds, cond in CELLS]
    cells.append((f"ds-mnist__cond-none__dist-{INFO}", "mnist/none"))

    labels, tv_prior, spread = [], [], []
    for exp, label in cells:
        paths = sorted(glob.glob(os.path.join(metrics_dir, exp + "__seed-*", "infonoise_profile.jsonl")))
        if not paths:
            continue
        per_seed = []
        centers = None
        for path in paths:
            r = last_run(path)[-1]
            centers = np.array(r["sigma_centers"])
            per_seed.append(np.array(r["pi"]))
        d_log = float(np.diff(np.log(centers)).mean())
        prior = warmup_prior_density(centers)
        tvs = [total_variation(pi, prior, d_log) for pi in per_seed]
        labels.append(f"{label}  (n={len(paths)})")
        tv_prior.append(float(np.mean(tvs)))
        spread.append(float(np.std(tvs)))

    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(labels) + 2.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, tv_prior, xerr=spread, color="#1f77b4", alpha=0.85, height=0.6, capsize=3)
    for yi, v in zip(y, tv_prior):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(1.0, color="black", lw=1)
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("total variation distance between learned pi and the warm-up prior")
    ax.set_title(
        "How far did InfoNoise move from logit_normal(0,1)?\n"
        "0 = never left it; 1 = no overlap. The baseline IS that prior,\n"
        "so small values mean the two arms train on nearly the same distribution.",
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/master_fid_results.cluster.json")
    ap.add_argument("--metrics_dir", default="logs/metrics")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = json.load(open(args.results))
    plot_forest(results, os.path.join(args.out_dir, "infonoise_forest.png"))
    plot_ranks(results, os.path.join(args.out_dir, "infonoise_ranks.png"))
    plot_curves(results, os.path.join(args.out_dir, "infonoise_curves.png"))
    plot_profiles(os.path.join(args.out_dir, "infonoise_profiles.png"), args.metrics_dir)
    plot_prior_distance(os.path.join(args.out_dir, "infonoise_prior_distance.png"), args.metrics_dir)


if __name__ == "__main__":
    main()
