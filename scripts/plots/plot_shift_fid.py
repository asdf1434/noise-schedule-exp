# WRITTEN BY CLAUDE

"""FID vs epoch, one line per training distribution, one panel per sampling schedule.

Complements scripts/plots/plot.py, which plots a SINGLE experiment (one training
distribution, one seed) with a curve per schedule. Here the comparison runs the
other way: the training distribution is what varies within a panel, because the
question the shifted-regime sweeps ask is which training distribution wins, with
the sampling schedule held fixed as a nuisance axis.

Seeds are aggregated rather than drawn individually -- mean with a +/- standard
error band across the 10 seeds per arm. The band matters as much as the line:
the InfoNoise arms have visibly tighter seed variance than the fixed ones, and
that is only readable if the spread is drawn.

    python scripts/plots/plot_shift_fid.py --dataset mnist_x10 --dataset mnist_x0.1

Writes into plots/infonoise_scaling/ by default. mnist carries 16 training
distributions from unrelated earlier sweeps, so it needs --dist to stay readable:

    python scripts/plots/plot_shift_fid.py --dataset mnist \
        --dist '^infonoise' --dist '^logit_normal_mu_0\.0_sigma_1\.0$' --dist '^uniform$' 
"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Experiments written before the ds-/cond-/dist-/seed- rename carry a bare name
# and are all unshifted mnist; see _belongs_to_dataset in evaluate_fid.py, which
# resolves them the same way.
LEGACY_RE = re.compile(r"^(?P<dist>.+?)_seed(?P<seed>\d+)$")
MODERN_RE = re.compile(
    r"^ds-(?P<ds>.+?)__cond-(?P<cond>[^_]+)__dist-(?P<dist>.+?)__seed-(?P<seed>\d+)$"
)


def _parse(exp_name: str):
    """-> (dataset, dist, seed), or None for names that fit neither scheme."""
    m = MODERN_RE.match(exp_name)
    if m:
        if m.group("cond") != "none":
            return None
        return m.group("ds"), m.group("dist"), int(m.group("seed"))
    m = LEGACY_RE.match(exp_name)
    if m and not exp_name.startswith("eurosat_"):
        return "mnist", m.group("dist"), int(m.group("seed"))
    return None


def _label(dist: str) -> str:
    """Short human label. mu is in t-space; sigma = (1-t)/t is what it means."""
    m = re.match(r"logit_normal_mu_(-?[\d.]+)_sigma_([\d.]+)$", dist)
    if m:
        mu, width = float(m.group(1)), float(m.group(2))
        centre = np.exp(-mu)  # noise level sigma=(1-t)/t at the distribution's centre
        return f"logit-normal(mu={mu:g}, s={width:g})  ~sigma {centre:.3g}"
    if dist.startswith("infonoise"):
        gate = re.search(r"gate_c_([\d.]+)", dist)
        return f"InfoNoise (gate_c={gate.group(1)})" if gate else "InfoNoise (auto gate)"
    return dist


def _collect(data: dict, dataset: str):
    """-> {dist: {schedule: {epoch: [fid per seed]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for exp_name, schedules in data.items():
        parsed = _parse(exp_name)
        if parsed is None or parsed[0] != dataset:
            continue
        _, dist, _seed = parsed
        for schedule, epochs in schedules.items():
            for epoch_key, score in epochs.items():
                digits = "".join(c for c in epoch_key if c.isdigit())
                if digits:
                    out[dist][schedule][int(digits)].append(score)
    return out


def plot_dataset(data: dict, dataset: str, save_dir: str, include=None):
    collected = _collect(data, dataset)
    if not collected:
        print(f"[skip] no experiments found for dataset={dataset}")
        return

    if include:
        pattern = re.compile("|".join(include))
        collected = {d: v for d, v in collected.items() if pattern.search(d)}
        if not collected:
            print(f"[skip] --dist filters matched nothing for dataset={dataset}")
            return

    schedules = sorted({s for d in collected.values() for s in d})
    dists = sorted(collected)

    ncols = 3
    nrows = -(-len(schedules) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False, sharex=True
    )
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, schedule in enumerate(schedules):
        ax = axes[idx // ncols][idx % ncols]
        for c_idx, dist in enumerate(dists):
            per_epoch = collected[dist].get(schedule)
            if not per_epoch:
                continue
            epochs = sorted(per_epoch)
            mean = np.array([np.mean(per_epoch[e]) for e in epochs])
            n = np.array([len(per_epoch[e]) for e in epochs])
            sem = np.array(
                [
                    np.std(per_epoch[e], ddof=1) / np.sqrt(len(per_epoch[e]))
                    if len(per_epoch[e]) > 1
                    else 0.0
                    for e in epochs
                ]
            )
            color = colors[c_idx % len(colors)]
            ax.plot(
                epochs, mean, marker="o", ms=3.5, lw=1.8, color=color,
                label=f"{_label(dist)}  (n={n.max()})",
            )
            ax.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
        ax.set_title(schedule, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("FID")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("epoch")

    for idx in range(len(schedules), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # Collect handles across panels: an arm missing from the first panel would
    # otherwise be absent from the legend entirely.
    seen, handles, labels = {}, [], []
    for row in axes:
        for ax in row:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    seen[lab] = True
                    handles.append(h)
                    labels.append(lab)
    ncol = 2 if len(labels) <= 6 else 3
    legend_rows = -(-len(labels) // ncol)
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )
    legend_frac = min(0.30, 0.028 * legend_rows + 0.03)
    fig.suptitle(
        f"{dataset}: FID vs epoch by training distribution "
        f"(mean +/- s.e.m. across seeds)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, legend_frac, 1, 0.97])

    os.makedirs(save_dir, exist_ok=True)
    suffix = "_filtered" if include else ""
    path = os.path.join(save_dir, f"fid_by_train_dist_{dataset}{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}  ({len(dists)} training dists x {len(schedules)} schedules)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", default="results/master_fid_results.json",
        help="master FID results written by evaluate_fid.py / merge_fid_shards.py",
    )
    parser.add_argument(
        "--dataset", action="append", required=True,
        help="dataset to plot; repeat for several",
    )
    parser.add_argument(
        "--dist", action="append",
        help="regex selecting which training distributions to draw; repeat to add "
             "more. Without it every distribution found for the dataset is drawn, "
             "which is unreadable for mnist (16 arms from unrelated sweeps).",
    )
    parser.add_argument("--save_dir", default="plots/infonoise_scaling")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    for dataset in args.dataset:
        plot_dataset(data, dataset, args.save_dir, include=args.dist)


if __name__ == "__main__":
    main()
