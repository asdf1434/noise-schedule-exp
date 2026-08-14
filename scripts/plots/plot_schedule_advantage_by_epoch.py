# WRITTEN BY CLAUDE

"""Does the sampling-schedule advantage change over training?

This is the "free" confound test from UPDATE_david_2026-07-29.md: the schedule
benefit (shifted_coarse vs uniform steps) shrinks and reverses as we move
MNIST -> Fashion-MNIST -> CIFAR-10 -> EuroSAT-64, but dataset complexity and
model underfit are confounded across those groups. Within a single group,
though, training epoch varies fit quality while holding the dataset fixed.

  - If the gradient is about FIT QUALITY, the advantage should track fit within
    a group: EuroSAT-64 (FID ~260 -> ~165 over training) should look more like
    the well-fit groups early and drift toward its reversal late (or vice
    versa) -- i.e. a strong epoch trend.
  - If it is about the DATA, each group's advantage should be roughly flat in
    epoch once past the first checkpoints.

Paired per (train_dist, seed): both schedules come from the same trained model,
so the comparison is within-model and the seed noise cancels. Pools over the
three "neutral" training distributions only (uniform / logit_normal(0,1) /
plateau), which the sweep found statistically indistinguishable -- the peaked
and skewed dists are excluded so a known-bad training distribution doesn't
drive the trend.

Run from the repo root:

    python -m scripts.plots.plot_schedule_advantage_by_epoch
    python -m scripts.plots.plot_schedule_advantage_by_epoch --schedule shifted_fine
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.getcwd())

from scripts.plots.plot_style import SCHEDULE_COLORS  # noqa: E402

# The three training distributions the sweep found indistinguishable.
NEUTRAL_DISTS = [
    "uniform",
    "logit_normal_mu_0.0_sigma_1.0",
    "plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
]

# Display order: roughly increasing dataset difficulty / absolute FID.
GROUP_ORDER = [
    "ds-mnist__cond-inpaint",
    "ds-fashion_mnist__cond-none",
    "ds-fashion_mnist__cond-inpaint",
    "ds-cifar10__cond-none",
    "ds-cifar10__cond-inpaint",
    "ds-eurosat64__cond-none",
]


def parse_key(key):
    """"ds-X__cond-Y__dist-Z__seed-N" -> (group, dist, seed); None for legacy keys."""
    if not key.startswith("ds-"):
        return None
    try:
        ds_part, cond_part, dist_part, seed_part = key.split("__")
        return (
            f"{ds_part}__{cond_part}",
            dist_part[len("dist-"):],
            int(seed_part[len("seed-"):]),
        )
    except ValueError:
        return None


def collect(results, schedule, baseline, epochs):
    """group -> {epoch: (deltas_pct, baseline_fids)}, paired per (dist, seed)."""
    per_group = {}
    for key, by_schedule in results.items():
        parsed = parse_key(key)
        if parsed is None:
            continue
        group, dist, _seed = parsed
        if dist not in NEUTRAL_DISTS:
            continue
        if schedule not in by_schedule or baseline not in by_schedule:
            continue
        for epoch in epochs:
            epoch_key = f"epoch_{epoch}"
            treat = by_schedule[schedule].get(epoch_key)
            base = by_schedule[baseline].get(epoch_key)
            if treat is None or base is None or base <= 0:
                continue
            slot = per_group.setdefault(group, {}).setdefault(epoch, ([], []))
            slot[0].append(100.0 * (treat - base) / base)
            slot[1].append(base)
    return per_group


def summarize(deltas):
    """Mean % delta, its 95% CI half-width, and a paired-test p-value."""
    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float("nan"), float("nan")
    sem = float(stats.sem(arr))
    half = sem * stats.t.ppf(0.975, arr.size - 1)
    p = float(stats.ttest_1samp(arr, 0.0).pvalue)
    return mean, half, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/master_fid_results.json")
    parser.add_argument("--schedule", default="shifted_coarse")
    parser.add_argument("--baseline", default="uniform")
    parser.add_argument("--epochs", default="10-100:10", help="start-end:step")
    parser.add_argument(
        "--fid_tol",
        type=float,
        default=0.05,
        help="two checkpoints count as matched-fit if their baseline FIDs are within this fraction",
    )
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    span, step = args.epochs.split(":")
    start, end = span.split("-")
    epochs = list(range(int(start), int(end) + 1, int(step)))

    with open(args.results) as f:
        results = json.load(f)

    per_group = collect(results, args.schedule, args.baseline, epochs)
    groups = [g for g in GROUP_ORDER if g in per_group]
    groups += sorted(g for g in per_group if g not in GROUP_ORDER)

    print(f"\n{args.schedule} vs {args.baseline} steps -- % FID change by epoch")
    print("(negative = schedule helps; paired per dist x seed, neutral train dists only)\n")

    fig, (ax_delta, ax_fid) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    trend_rows = []
    for i, group in enumerate(groups):
        by_epoch = per_group[group]
        present = [e for e in epochs if e in by_epoch]
        means, halves, ps, base_fids, ns = [], [], [], [], []
        for epoch in present:
            deltas, bases = by_epoch[epoch]
            mean, half, p = summarize(deltas)
            means.append(mean)
            halves.append(half)
            ps.append(p)
            base_fids.append(float(np.mean(bases)))
            ns.append(len(deltas))

        label = group.replace("ds-", "").replace("__cond-", " / ")
        print(f"{label}")
        print(f"  {'epoch':>6} {'delta%':>9} {'95% CI':>16} {'p':>10} {'base FID':>9} {'n':>5}")
        for epoch, mean, half, p, base, n in zip(present, means, halves, ps, base_fids, ns):
            ci = f"[{mean - half:+6.1f},{mean + half:+6.1f}]"
            print(f"  {epoch:>6} {mean:>+9.1f} {ci:>16} {p:>10.2e} {base:>9.1f} {n:>5}")

        # Is the advantage trending with epoch, and does it track fit quality?
        if len(present) >= 3:
            slope, _, r, p_trend, _ = stats.linregress(present, means)
            r_fit, p_fit = stats.pearsonr(base_fids, means)
            trend_rows.append((label, slope, p_trend, r, r_fit, p_fit, means[0], means[-1]))

        color = color_cycle[i % len(color_cycle)]
        ax_delta.errorbar(
            present, means, yerr=halves, marker="o", capsize=3, color=color, label=label
        )
        ax_fid.plot(present, base_fids, marker="o", color=color, label=label)
        print()

    print("Trend of the advantage across training (per group):")
    print(
        f"  {'group':<28} {'slope %/epoch':>14} {'p':>10} {'r(epoch)':>9} "
        f"{'r(baseFID)':>11} {'p':>10} {'ep10':>8} {'ep100':>8}"
    )
    for label, slope, p_trend, r, r_fit, p_fit, first, last in trend_rows:
        print(
            f"  {label:<28} {slope:>+14.3f} {p_trend:>10.2e} {r:>+9.2f} "
            f"{r_fit:>+11.2f} {p_fit:>10.2e} {first:>+8.1f} {last:>+8.1f}"
        )
    print()

    # The sharpest form of the argument: find pairs of (group, epoch) checkpoints
    # from *different* datasets whose baseline FID nearly matches. Fit quality is
    # then held fixed by construction, so any surviving difference in the schedule
    # advantage has to come from the data rather than from how well it is modeled.
    print(f"Matched fit quality (baseline FID within {args.fid_tol:.0%}, different datasets):")
    points = []
    for group in groups:
        for epoch, (deltas, bases) in per_group[group].items():
            label = group.replace("ds-", "").replace("__cond-", " / ")
            points.append((label, group.split("__")[0], epoch, float(np.mean(bases)), float(np.mean(deltas))))

    shown = set()
    for a in sorted(points, key=lambda p: p[3]):
        for b in sorted(points, key=lambda p: p[3]):
            if a[1] == b[1] or abs(a[3] - b[3]) > args.fid_tol * min(a[3], b[3]):
                continue
            pair = tuple(sorted([(a[0], a[2]), (b[0], b[2])]))
            if pair in shown:
                continue
            shown.add(pair)
            print(
                f"  {a[0]:<24} ep{a[2]:<4} FID {a[3]:6.1f}  ->  {a[4]:+6.1f}%   vs   "
                f"{b[0]:<24} ep{b[2]:<4} FID {b[3]:6.1f}  ->  {b[4]:+6.1f}%"
                f"   (gap {abs(a[4] - b[4]):.1f} pts)"
            )
    if not shown:
        print("  (no cross-dataset checkpoints matched within tolerance)")
    print()

    ax_delta.axhline(0, color="0.4", lw=1, ls="--")
    ax_delta.set_ylabel(f"FID change vs {args.baseline} steps (%)")
    ax_delta.set_title(
        f"Is the {args.schedule} advantage a dataset effect or a fit-quality effect?\n"
        "flat in epoch = dataset; tracks FID = fit quality"
    )
    ax_delta.legend(fontsize=8)
    ax_delta.grid(alpha=0.3)

    ax_fid.set_yscale("log")
    ax_fid.set_ylabel(f"{args.baseline}-steps FID\n(fit quality)")
    ax_fid.set_xlabel("epoch")
    ax_fid.grid(alpha=0.3)

    fig.tight_layout()
    save_path = args.save_path or f"plots/schedule_advantage_by_epoch_{args.schedule}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
