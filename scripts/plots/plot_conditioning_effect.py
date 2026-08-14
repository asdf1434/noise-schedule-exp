# WRITTEN BY CLAUDE

"""Charts for the inpaint-vs-unconditional comparison and the schedule gradient.

Reads results/master_fid_results.json and writes three PNGs into plots/:

  conditioning_effect_dumbbell.png  all 24 (train dist x sampling schedule) cells
                                    per dataset, unconditional -> inpaint
  ..._dumbbell_ci.png               the same, with 95% CIs of each mean
  conditioning_gain_by_schedule.png % FID reduction from conditioning, per
                                    sampling schedule (the substitution finding)
  schedule_gradient.png             shifted_coarse vs uniform steps, per dataset

mnist, fashion_mnist and cifar10 have both conditioning arms, so the conditioning
figures cover those three (mnist's unconditional arm comes from the pre-rename
exp1/exp2 sweep -- see load()); schedule_gradient.png covers all six
dataset/conditioning groups, including eurosat64.

Colors come from the validated categorical palette (slots 1-2) rather than
matplotlib's default cycle, which isn't CVD-checked -- see plot_style.py for the
older per-dist/per-schedule scheme used by the FID-curve plots.

Usage: python scripts/plots/plot_conditioning_effect.py
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = "results/master_fid_results.json"
OUT_DIR = "plots"
EPOCH = 100

# Validated categorical slots 1 and 2 (light mode), plus chrome/ink roles.
C_NONE = "#2a78d6"      # slot 1, blue
C_INPAINT = "#eb6834"   # slot 2, orange
C_SLOT3 = "#1baf7a"     # slot 3, aqua (sub-3:1 on light -> every bar is labelled)
C_BETTER = "#2a78d6"    # diverging cool pole
C_WORSE = "#e34948"     # diverging warm pole
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SCHEDULES = ["uniform", "logit_normal", "shifted_fine", "shifted_coarse"]
SCHEDULE_LABELS = {
    "uniform": "uniform",
    "logit_normal": "logit-normal CDF",
    "shifted_fine": "shifted fine",
    "shifted_coarse": "shifted coarse",
}
DISTS = [
    "uniform",
    "logit_normal_mu_0.0_sigma_1.0",
    "plateau_logit_normal_mu_0.0_sigma_1.0_uniform_prob_0.3",
    "logit_normal_mu_0.0_sigma_0.3",
    "logit_normal_mu_-1.5_sigma_1.0",
    "logit_normal_mu_1.5_sigma_1.0",
]
DIST_LABELS = dict(
    zip(
        DISTS,
        ["uniform", "ln(0, 1.0)", "plateau", "ln(0, 0.3)", "ln(-1.5, 1)", "ln(+1.5, 1)"],
    )
)
NEUTRAL_DISTS = DISTS[:3]
# Datasets with both conditioning arms. mnist's "none" arm is the pre-rename
# exp1/exp2 sweep -- see load().
COND_DATASETS = ["mnist", "fashion_mnist", "cifar10"]
# The legacy uniform-dist arm only ran 5 seeds; flagged in the figures.
LEGACY_THIN = "mnist unconditional = pre-rename exp1/exp2 runs (uniform dist: 5 seeds, not 20)"
GROUPS = [
    ("mnist", "inpaint"),
    ("fashion_mnist", "none"),
    ("fashion_mnist", "inpaint"),
    ("cifar10", "none"),
    ("cifar10", "inpaint"),
    ("eurosat64", "none"),
]


def load():
    """(dataset, cond, dist, seed, schedule, epoch) -> FID, seeds 0-19 only.

    MNIST's unconditional arm predates the ds-/cond-/dist- rename, so the
    pre-rename keys ("<dist>_seed<n>", no dataset prefix) are folded in as
    ("mnist", "none", ...). That's sound rather than a fudge: since the exp1-era
    commit, src/loss.py is unchanged, src/model.py differs only in comments and
    jaxtyping annotations, the Euler sampler and schedules are untouched, and
    MNIST's [-1, 1] normalization survived the datasets.py refactor unchanged --
    so the two arms differ only by --conditioning. The one caveat is coverage:
    the legacy `uniform` training dist has 5 seeds, not 20 (see LEGACY_THIN).
    """
    with open(RESULTS) as f:
        raw = json.load(f)
    table = {}
    for name, per_schedule in raw.items():
        m = re.match(r"ds-(.+?)__cond-(.+?)__dist-(.+?)__seed-(\d+)$", name)
        if m:
            dataset, cond = m.group(1), m.group(2)
            dist, seed = m.group(3), int(m.group(4))
        else:
            legacy = re.match(r"(.+)_seed(\d+)$", name)
            if not legacy or name.startswith("eurosat_"):
                continue
            dataset, cond = "mnist", "none"
            dist, seed = legacy.group(1), int(legacy.group(2))
            if dist not in DISTS:  # exp3 sigma-sweep dists aren't in the 6-dist grid
                continue
        if seed >= 20:
            continue
        for schedule, epochs in per_schedule.items():
            for epoch, score in epochs.items():
                key = (dataset, cond, dist, seed, schedule, int(epoch.split("_")[1]))
                table[key] = score
    return table


def cell_values(table, dataset, cond, dist, schedule):
    """Per-seed FIDs at EPOCH for one cell."""
    return [
        table[(dataset, cond, dist, s, schedule, EPOCH)]
        for s in range(20)
        if (dataset, cond, dist, s, schedule, EPOCH) in table
    ]


def cell(table, dataset, cond, dist, schedule):
    """Mean FID over seeds at EPOCH, or None if the cell is empty."""
    vals = cell_values(table, dataset, cond, dist, schedule)
    return float(np.mean(vals)) if vals else None


def cell_ci(table, dataset, cond, dist, schedule):
    """(mean, half-width of the 95% CI of the mean, n) or None.

    CI of the mean rather than +/-1 SD: the chart's job is "is the gap between
    the two arms real", not "how much do seeds vary". With n=20 most intervals
    are invisibly small, which is the honest read -- the gaps dwarf them.
    """
    vals = cell_values(table, dataset, cond, dist, schedule)
    if not vals:
        return None
    n = len(vals)
    if n < 2:
        return float(vals[0]), 0.0, n
    half = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(n)
    return float(np.mean(vals)), half, n


def style_axes(ax, xgrid=True, ygrid=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_dumbbell(table):
    """24 cells per dataset: unconditional -> inpaint, as paired dots."""
    datasets = COND_DATASETS
    rows = [(d, s) for d in DISTS for s in SCHEDULES]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), facecolor=SURFACE)
    for ax, dataset in zip(axes, datasets):
        style_axes(ax)
        ys, deltas = [], []
        for i, (dist, schedule) in enumerate(rows):
            y = len(rows) - 1 - i
            a = cell(table, dataset, "none", dist, schedule)
            b = cell(table, dataset, "inpaint", dist, schedule)
            if a is None or b is None:
                continue
            ys.append(y)
            deltas.append((y, a, b, 100.0 * (b - a) / a))
            ax.plot([a, b], [y, y], color=BASELINE, linewidth=2, zorder=1,
                    solid_capstyle="round")
            # 2px surface ring keeps the two dots readable where they nearly touch
            ax.plot(a, y, "o", markersize=7, color=C_NONE, zorder=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot(b, y, "o", markersize=7, color=C_INPAINT, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)

        # Direct-label only the extremes, per the selective-labelling rule.
        # Labels sit to the RIGHT of the higher dot -- to the left they collide
        # with the y tick labels.
        if deltas:
            best = min(deltas, key=lambda t: t[3])
            worst = max(deltas, key=lambda t: t[3])
            for y, a, b, pct in {id(best): best, id(worst): worst}.values():
                ax.annotate(
                    f"{pct:+.0f}%",
                    xy=(max(a, b), y),
                    xytext=(9, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=INK_2,
                    fontweight="bold",
                )

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{DIST_LABELS[d]}  ·  {SCHEDULE_LABELS[s]}" for d, s in rows][::-1],
            fontsize=7.5,
            color=INK_2,
        )
        ax.set_ylim(-1, len(rows))
        ax.set_xlabel("FID at epoch 100  (lower is better)", fontsize=8.5, color=INK_2)
        title = dataset + ("  *" if dataset == "mnist" else "")
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")

        # Hairline separators between train-dist blocks
        for k in range(1, len(DISTS)):
            ax.axhline(len(rows) - k * len(SCHEDULES) - 0.5, color=GRID, linewidth=0.8, zorder=0)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=C_NONE,
                   label="unconditional"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=C_INPAINT,
                   label="inpaint conditioning"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.978),
               frameon=False, fontsize=9, labelcolor=INK_2, ncol=2)
    fig.text(0.006, 0.012, "* " + LEGACY_THIN, fontsize=8, color=MUTED)
    fig.suptitle(
        "Inpaint conditioning improves every training-distribution x sampling-schedule cell",
        fontsize=13, color=INK, fontweight="bold", x=0.008, ha="left", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def fig_dumbbell_ci(table):
    """Same as fig_dumbbell, with 95% CI of the mean on each dot."""
    datasets = COND_DATASETS
    rows = [(d, s) for d in DISTS for s in SCHEDULES]
    thin_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), facecolor=SURFACE)
    for ax, dataset in zip(axes, datasets):
        style_axes(ax)
        for i, (dist, schedule) in enumerate(rows):
            y = len(rows) - 1 - i
            left = cell_ci(table, dataset, "none", dist, schedule)
            right = cell_ci(table, dataset, "inpaint", dist, schedule)
            if left is None or right is None:
                continue
            ax.plot([left[0], right[0]], [y, y], color=BASELINE, linewidth=2, zorder=1,
                    solid_capstyle="round")
            for (mean, half, n), color, z in ((left, C_NONE, 2), (right, C_INPAINT, 3)):
                if half > 0:
                    # Whisker in the series colour reads as an extension of the dot
                    ax.plot([mean - half, mean + half], [y, y], color=color,
                            linewidth=1.4, zorder=z, solid_capstyle="butt")
                ax.plot(mean, y, "o", markersize=6, color=color, zorder=z + 3,
                        markeredgecolor=SURFACE, markeredgewidth=1.4)
                if n < 20:
                    thin_rows.append((dataset, dist, schedule, n))

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{DIST_LABELS[d]}  ·  {SCHEDULE_LABELS[s]}" for d, s in rows][::-1],
            fontsize=7.5, color=INK_2,
        )
        ax.set_ylim(-1, len(rows))
        ax.set_xlabel("FID at epoch 100  (lower is better)", fontsize=8.5, color=INK_2)
        title = dataset + ("  *" if dataset == "mnist" else "")
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")
        for k in range(1, len(DISTS)):
            ax.axhline(len(rows) - k * len(SCHEDULES) - 0.5, color=GRID, linewidth=0.8,
                       zorder=0)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=C_NONE,
                   label="unconditional"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=C_INPAINT,
                   label="inpaint conditioning"),
        plt.Line2D([], [], linestyle="-", linewidth=1.4, color=MUTED,
                   label="95% CI of the mean"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.978),
               frameon=False, fontsize=9, labelcolor=INK_2, ncol=3)
    fig.suptitle(
        "Every cell improves by far more than the uncertainty on either mean",
        fontsize=13, color=INK, fontweight="bold", x=0.006, ha="left", y=0.995,
    )
    note = "20 seeds per cell.  * " + LEGACY_THIN
    if thin_rows:
        note += f"  ({len(thin_rows)} cells below 20 seeds, drawn with wider intervals)"
    fig.text(0.006, 0.012, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    return fig


def fig_gain_by_schedule(table):
    """% FID reduction from conditioning, per schedule, neutral dists pooled."""
    datasets = COND_DATASETS
    colors = {"mnist": C_NONE, "fashion_mnist": C_INPAINT, "cifar10": C_SLOT3}

    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    style_axes(ax, xgrid=False, ygrid=True)
    width = 0.22
    x = np.arange(len(SCHEDULES))

    lo = 0.0
    for j, dataset in enumerate(datasets):
        gains = []
        for schedule in SCHEDULES:
            a = np.mean([cell(table, dataset, "none", d, schedule) for d in NEUTRAL_DISTS])
            b = np.mean([cell(table, dataset, "inpaint", d, schedule) for d in NEUTRAL_DISTS])
            gains.append(100.0 * (b - a) / a)
        lo = min(lo, min(gains))
        # 2px surface gap between adjacent bars
        offset = (j - 1) * (width + 0.03)
        label = dataset + (" *" if dataset == "mnist" else "")
        bars = ax.bar(x + offset, gains, width, color=colors[dataset], label=label,
                      linewidth=0)
        for rect, gain in zip(bars, gains):
            ax.annotate(f"{gain:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, gain),
                        xytext=(0, -4), textcoords="offset points", ha="center",
                        va="top", fontsize=8, color=INK_2, fontweight="bold")

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_ylim(lo * 1.16, 0)
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEDULE_LABELS[s] for s in SCHEDULES], fontsize=9, color=INK_2)
    ax.set_ylabel("% change in FID from adding inpaint conditioning", fontsize=8.5, color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left",
              bbox_to_anchor=(0, 1.06), ncol=3)
    ax.set_title(
        "On the MNIST-like datasets the coarse schedule gains least from conditioning; on cifar10 the gain is flat",
        fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=30,
    )
    fig.text(0.012, 0.005, "* " + LEGACY_THIN, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_schedule_gradient(table):
    """shifted_coarse vs uniform steps, per dataset/conditioning group.

    Non-significant groups are drawn in muted gray: painting cifar10's +1.4%
    (p=0.14) in the same "worse" red as eurosat64's +6.0% (p=8e-9) would assert
    an effect the data doesn't support.
    """
    from scipy import stats

    labels, values, pvals = [], [], []
    for dataset, cond in GROUPS:
        a, b = [], []
        for dist in NEUTRAL_DISTS:
            for s in range(20):
                keys = [(dataset, cond, dist, s, sc, EPOCH) for sc in SCHEDULES]
                if all(k in table for k in keys):
                    a.append(table[(dataset, cond, dist, s, "shifted_coarse", EPOCH)])
                    b.append(table[(dataset, cond, dist, s, "uniform", EPOCH)])
        labels.append(f"{dataset}\n{cond}")
        values.append(100.0 * (np.mean(a) - np.mean(b)) / np.mean(b))
        pvals.append(stats.wilcoxon(np.array(a), np.array(b)).pvalue)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    style_axes(ax, xgrid=False, ygrid=True)
    colors = [
        BASELINE if p >= 0.05 else (C_WORSE if v > 0 else C_BETTER)
        for v, p in zip(values, pvals)
    ]
    bars = ax.bar(range(len(values)), values, 0.5, color=colors, linewidth=0)
    for rect, v, p in zip(bars, values, pvals):
        tag = f"{v:+.1f}%" + ("  n.s." if p >= 0.05 else "")
        ax.annotate(tag, xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 4 if v > 0 else -4), textcoords="offset points",
                    ha="center", va="bottom" if v > 0 else "top", fontsize=9,
                    color=INK_2, fontweight="bold")

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK_2)
    ax.set_ylim(min(values) * 1.15, max(values) * 2.2)
    ax.set_ylabel(
        "% change in FID vs uniform steps\n(negative = shifted coarse is better)",
        fontsize=8.5, color=INK_2,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_BETTER, label="coarse better (p < 0.05)"),
        plt.Rectangle((0, 0), 1, 1, color=C_WORSE, label="coarse worse (p < 0.05)"),
        plt.Rectangle((0, 0), 1, 1, color=BASELINE, label="no significant difference"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK_2,
              loc="upper left", bbox_to_anchor=(0, 1.07), ncol=3)
    ax.set_title(
        "The shifted-coarse sampling schedule helps on MNIST-like data and reverses on EuroSAT-64",
        fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=30,
    )
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    table = load()
    for name, builder in [
        ("conditioning_effect_dumbbell", fig_dumbbell),
        ("conditioning_effect_dumbbell_ci", fig_dumbbell_ci),
        ("conditioning_gain_by_schedule", fig_gain_by_schedule),
        ("schedule_gradient", fig_schedule_gradient),
    ]:
        fig = builder(table)
        path = os.path.join(OUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
