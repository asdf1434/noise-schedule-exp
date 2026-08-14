# WRITTEN BY CLAUDE

"""How much fine detail does each dataset have, and does that predict which step
spacing wins?

The story so far is that the best sampling-step spacing differs by dataset:
`shifted_coarse` is worth -57% FID on MNIST, nothing on CIFAR-10, and is
actively worse on EuroSAT-64. If that tracks something measurable about the
images, it stops being a catalogue of results and becomes a rule you can apply
to a dataset you haven't trained on yet.

The obvious candidate is how quickly the power spectrum falls off with spatial
frequency -- a steep slope means the image is dominated by large smooth
structure, a shallow one means lots of fine texture.

**Run this BEFORE the new runs finish.** It fits the relationship on the six
groups that already have results, then writes down a predicted advantage for
each new dataset. Fitting first is the whole point: a line fitted afterwards to
all the data would prove nothing, whereas a prediction made in advance and then
checked is a real test. Re-run it after the results land to compare.

    python -m scripts.plots.plot_spectral_slope
    python -m scripts.plots.plot_spectral_slope --n_images 5000

Also reports a sanity check on FID itself. The 14x14 arm is scored by upscaling
to Inception's 299x299, and if FID discriminates less well at that size, a
smaller measured advantage there could be a measurement artefact rather than a
real effect. The spread of FID between the first and last checkpoint is a rough
proxy for how much room the metric has to work with, so a collapse at 14x14 is
a warning that the arm can't be read the way we intend.
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

from scripts.plots.plot_schedule_advantage_by_epoch import (  # noqa: E402
    collect,
    parse_key,
)
from src.datasets import DATASETS  # noqa: E402

# Groups that already have results, and the dataset each one's images come from.
FITTED_GROUPS = {
    "ds-mnist__cond-inpaint": "mnist",
    "ds-fashion_mnist__cond-none": "fashion_mnist",
    "ds-fashion_mnist__cond-inpaint": "fashion_mnist",
    "ds-cifar10__cond-none": "cifar10",
    "ds-cifar10__cond-inpaint": "cifar10",
    "ds-eurosat64__cond-none": "eurosat64",
}

# Datasets the new sweeps introduce, which the fit above is used to predict.
HELD_OUT = [
    "mnist_g050",
    "mnist_g100",
    "mnist_g250",
    "fashion_mnist_r14",
    "fashion_mnist_r28",
    "fashion_mnist_r56",
]


def load_real_images(dataset, n_images):
    """Read a dataset's FID reference images off disk as (N, H, W) in [0, 1].

    Deliberately reads the PNGs that evaluate_fid.py already scores against
    rather than calling DATASETS[name].load(): the loaders pull the entire
    60,000-image split (and will download it if it isn't cached) just to hand
    back the couple of thousand images needed here. Those folders are written by
    src/generate_real_samples.py, so run the prep job before this script.
    """
    from PIL import Image

    real_dir = DATASETS[dataset].real_dir
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(real_dir)
    files = sorted(f for f in os.listdir(real_dir) if f.endswith(".png"))[:n_images]
    if not files:
        raise FileNotFoundError(f"{real_dir} (no .png files)")
    # save_images writes greyscale triplicated across RGB; take one channel back.
    return np.stack(
        [np.asarray(Image.open(os.path.join(real_dir, f)), dtype=np.float32)[..., 0]
         for f in files]
    ) / 255.0


def radial_power_slope(images):
    """Slope of log(power) vs log(spatial frequency), radially averaged.

    ``images`` is (N, H, W). Scaling the images doesn't move the slope (it
    multiplies every frequency's power by the same constant), so this is
    unaffected by the per-level rescaling in _gamma_transform -- what it sees is
    the change in shape, which is what we want.
    """
    a = np.asarray(images)
    a = a - a.mean()
    power = (np.abs(np.fft.fftshift(np.fft.fft2(a), axes=(1, 2))) ** 2).mean(0)

    h, w = power.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(int)

    # Skip k=0 (the DC term, already removed) and stop at the Nyquist radius,
    # past which the circles run off the corners of the square and the average
    # is taken over progressively fewer, biased pixels.
    ks = np.arange(1, h // 2)
    profile = np.array([power[radius == k].mean() for k in ks])
    ok = profile > 0
    return float(np.polyfit(np.log(ks[ok]), np.log(profile[ok]), 1)[0])


def advantage(results, group, schedule, baseline, epoch):
    """Mean % FID change vs the baseline spacing at one checkpoint, or None."""
    per_group = collect(results, schedule, baseline, [epoch])
    if group not in per_group or epoch not in per_group[group]:
        return None, 0
    deltas, _ = per_group[group][epoch]
    return (float(np.mean(deltas)), len(deltas)) if deltas else (None, 0)


def fid_dynamic_range(results, group, baseline, epochs):
    """First-to-last checkpoint FID spread -- a proxy for how much room the
    metric has to tell good samples from bad in this group."""
    per_group = collect(results, baseline, baseline, epochs)
    if group not in per_group:
        return None
    by_epoch = per_group[group]
    present = [e for e in epochs if e in by_epoch]
    if len(present) < 2:
        return None
    first = float(np.mean(by_epoch[present[0]][1]))
    last = float(np.mean(by_epoch[present[-1]][1]))
    return first, last, (first - last) / first * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/master_fid_results.json")
    parser.add_argument("--schedule", default="shifted_coarse")
    parser.add_argument("--baseline", default="uniform")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--epochs", default="10-100:10", help="start-end:step")
    parser.add_argument(
        "--n_images",
        type=int,
        default=2000,
        help="images per dataset for the spectrum estimate",
    )
    parser.add_argument("--out", default="results/spectral_slopes.json")
    parser.add_argument("--save_path", default="plots/spectral_slope_fit.png")
    args = parser.parse_args()

    span, step = args.epochs.split(":")
    start, end = span.split("-")
    epochs = list(range(int(start), int(end) + 1, int(step)))

    with open(args.results) as f:
        results = json.load(f)

    wanted = sorted(set(FITTED_GROUPS.values()) | set(HELD_OUT))
    slopes, missing = {}, []
    print("Measuring how fast the power spectrum falls off (steeper = less fine detail)\n")
    for name in wanted:
        if name not in DATASETS:
            print(f"  {name:>20}  SKIPPED (not in the registry)")
            continue
        try:
            images = load_real_images(name, args.n_images)
        except FileNotFoundError as exc:
            missing.append(name)
            print(f"  {name:>20}  SKIPPED (no reference images yet: {exc})")
            continue
        slopes[name] = radial_power_slope(images)
        print(f"  {name:>20}  slope {slopes[name]:>7.3f}   "
              f"({images.shape[1]}px, {len(images)} images)")
        del images

    if missing:
        print(f"\n  {len(missing)} dataset(s) have no reference images yet. Run"
              f"\n    sbatch scripts/slurm/run_overnight_prep.sh"
              f"\n  (or python -m src.generate_real_samples --dataset <name>) first.")

    # --- fit on the groups that already have results -------------------------
    xs, ys, labels = [], [], []
    for group, dataset in FITTED_GROUPS.items():
        if dataset not in slopes:
            continue
        adv, n = advantage(results, group, args.schedule, args.baseline, args.epoch)
        if adv is None:
            continue
        xs.append(slopes[dataset])
        ys.append(adv)
        labels.append(group.replace("ds-", "").replace("__cond-", " / "))

    print(f"\n{args.schedule} vs {args.baseline} at epoch {args.epoch}, "
          f"against spectrum slope:\n")
    print(f"  {'group':<28} {'slope':>8} {'advantage':>11}")
    for x, y, lab in sorted(zip(xs, ys, labels)):
        print(f"  {lab:<28} {x:>8.3f} {y:>+10.1f}%")

    payload = {"slopes": slopes, "fitted": dict(zip(labels, zip(xs, ys)))}

    if len(xs) >= 3:
        fit = stats.linregress(xs, ys)
        print(f"\n  fit: advantage = {fit.slope:.1f} * slope + {fit.intercept:.1f}")
        print(f"       r = {fit.rvalue:+.3f}, p = {fit.pvalue:.3g}, n = {len(xs)}")
        if len(set(xs)) < 4:
            print(f"       NOTE only {len(set(xs))} distinct datasets behind these "
                  f"{len(xs)} points -- treat the fit as indicative, not established.")

        print("\nPredicted advantage for the new datasets (recorded before results land):\n")
        print(f"  {'dataset':>20} {'slope':>8} {'predicted':>11}")
        preds = {}
        for name in HELD_OUT:
            if name not in slopes:
                continue
            pred = fit.slope * slopes[name] + fit.intercept
            preds[name] = {"slope": slopes[name], "predicted_advantage_pct": pred}
            print(f"  {name:>20} {slopes[name]:>8.3f} {pred:>+10.1f}%")
        payload["fit"] = {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r": fit.rvalue,
            "p": fit.pvalue,
            "n": len(xs),
        }
        payload["predictions"] = preds
    else:
        fit = None
        print("\n  Not enough groups with results to fit anything yet.")

    # --- is FID still discriminating at every size? --------------------------
    print("\nFID spread from first to last checkpoint (a metric-sensitivity check;")
    print("a collapse here means measured differences shrink for measurement reasons):\n")
    print(f"  {'group':<28} {'ep10':>9} {'ep100':>9} {'drop':>8}")
    ranges = {}
    for group in list(FITTED_GROUPS) + [
        f"ds-{d}__cond-none" for d in HELD_OUT
    ]:
        rng = fid_dynamic_range(results, group, args.baseline, epochs)
        if rng is None:
            continue
        first, last, pct = rng
        ranges[group] = {"first": first, "last": last, "drop_pct": pct}
        lab = group.replace("ds-", "").replace("__cond-", " / ")
        print(f"  {lab:<28} {first:>9.1f} {last:>9.1f} {pct:>7.1f}%")
    payload["fid_dynamic_range"] = ranges

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")

    # --- plot ----------------------------------------------------------------
    if xs:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.axhline(0, color="0.4", lw=1, ls="--")
        ax.scatter(xs, ys, s=60, zorder=3, label="groups with results (fitted)")
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        if fit is not None:
            held = [(slopes[n], n) for n in HELD_OUT if n in slopes]
            grid = np.linspace(
                min(xs + [h[0] for h in held]), max(xs + [h[0] for h in held]), 50
            )
            ax.plot(grid, fit.slope * grid + fit.intercept, color="0.5", lw=1,
                    label=f"fit (r={fit.rvalue:+.2f})")
            if held:
                hx = [h[0] for h in held]
                hy = [fit.slope * x + fit.intercept for x in hx]
                ax.scatter(hx, hy, marker="x", s=50, zorder=3,
                           label="new datasets (predicted)")
                for x, y, (_, n) in zip(hx, hy, held):
                    ax.annotate(n, (x, y), fontsize=6, xytext=(4, -8),
                                textcoords="offset points", color="0.35")
        ax.set_xlabel("power spectrum slope  (steeper / less fine detail  <--  -->  shallower)")
        ax.set_ylabel(f"{args.schedule} vs {args.baseline} steps (% FID)")
        ax.set_title("Does how much fine detail a dataset has predict\nwhich step spacing wins?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        fig.savefig(args.save_path, dpi=150)
        print(f"wrote {args.save_path}")


if __name__ == "__main__":
    main()
