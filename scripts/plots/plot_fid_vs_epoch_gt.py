"""FID over training for the gt_target runs against InfoNoise and the default
logit-normal(0, 1), one figure per cell.

Each arm is shown at its own best sampling schedule: the one with the lowest
mean FID over epochs 80-100. Lines are the mean over seeds, bands +-1 SD.

    python scripts/plots/plot_fid_vs_epoch_gt.py   # -> plots/sept30/fid_vs_epoch_<cell>.png, one per cell
    python scripts/plots/plot_fid_vs_epoch_gt.py --fid_json <merged.json> --note "Partial: 13 of 16 FID shards scored"
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (sigma_min, InfoNoise run name, t_clip suffix), matching infonoise_gridfix.conf
CELLS = {
    "mnist_x0.1": ("0.0002", "infonoise_sigma_min_0.0002_sigma_max_80.0_tclip_0.005", "_tclip_0.005"),
    "mnist": ("0.002", "infonoise", ""),
    "mnist_x10": ("0.02", "infonoise_sigma_min_0.02_sigma_max_80.0", ""),
    "cifar10_x0.1": ("0.0002", "infonoise_sigma_min_0.0002_sigma_max_80.0_tclip_0.005", "_tclip_0.005"),
    "cifar10": ("0.002", "infonoise", ""),
    "cifar10_x10": ("0.02", "infonoise_sigma_min_0.02_sigma_max_80.0", ""),
}
EPOCHS = list(range(10, 101, 10))
FINAL = ["epoch_80", "epoch_90", "epoch_100"]


def arms(ds):
    smin, info, tc = CELLS[ds]
    return [
        ("gt", "#2a6fdb", f"ds-{ds}__cond-none__dist-gt_target_sigma_min_{smin}_sigma_max_80.0{tc}"),
        (
            "gt per class",
            "#009e73",
            f"ds-{ds}__cond-class__dist-gt_target_sigma_min_{smin}_sigma_max_80.0_per_class_True{tc}",
        ),
        ("InfoNoise", "#e8710a", f"ds-{ds}__cond-none__dist-{info}"),
        ("default", "#6b7280", f"ds-{ds}__cond-none__dist-logit_normal_mu_0.0_sigma_1.0"),
    ]


def runs_of(fid, prefix):
    return [v for k, v in fid.items() if k.startswith(prefix + "__seed-")]


def best_schedule(runs):
    scores = {}
    for sched in {s for r in runs for s in r}:
        vals = [np.mean([r[sched][e] for e in FINAL]) for r in runs if sched in r and all(e in r[sched] for e in FINAL)]
        if vals:
            scores[sched] = np.mean(vals)
    return min(scores, key=scores.get) if scores else None


p = argparse.ArgumentParser()
p.add_argument("--fid_json", default="results/master_fid_results.json")
p.add_argument("--note", default="")
p.add_argument("--out_dir", default="plots/sept30")
args = p.parse_args()

with open(args.fid_json) as f:
    fid = json.load(f)
os.makedirs(args.out_dir, exist_ok=True)

caption = (
    "gt = trained on the B.4 target ρ*/w; gt per class = class-conditional model,\n"
    "each example's own class target; default = logit-normal(0, 1).\n"
    "Each arm at its best sampling schedule (in brackets). Mean over seeds, band ±1 SD."
)
if args.note:
    caption = f"{args.note}.\n" + caption

for ds in CELLS:
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for label, color, prefix in arms(ds):
        runs = runs_of(fid, prefix)
        sched = best_schedule(runs)
        if sched is None:
            continue
        mean, sd, xs = [], [], []
        for e in EPOCHS:
            vals = [r[sched][f"epoch_{e}"] for r in runs if f"epoch_{e}" in r.get(sched, {})]
            if vals:
                xs.append(e)
                mean.append(np.mean(vals))
                sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        mean, sd = np.array(mean), np.array(sd)
        ax.plot(xs, mean, color=color, linewidth=2, marker="o", markersize=3, label=f"{label} [{sched}]")
        ax.fill_between(xs, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
    base, _, scale = ds.partition("_")
    ax.set_title(f"{base} {scale or 'x1'}: FID over training")
    ax.set_xlabel("epoch")
    ax.set_ylabel("FID (log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    note = caption
    if "_x0.1" in ds:
        note += "\nThe default here used t_clip 0.05; the other arms 0.005."
    fig.text(0.0, -0.02, note, ha="left", va="top", fontsize=8)
    out = os.path.join(args.out_dir, f"fid_vs_epoch_{ds}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
