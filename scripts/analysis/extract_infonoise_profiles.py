"""Pull the per-cell InfoNoise profiles out of logs/metrics/ into one JSON.

Run on the cluster (that is where logs/metrics/ actually lives), then scp the
output back. Emits, per cell, the median across seeds of the final refresh's
m_hat and pi, plus the first refresh's pi and the grid bounds -- everything the
plotting scripts need, without copying 10 seeds x N refreshes of jsonl.

    python scripts/analysis/extract_infonoise_profiles.py --out /tmp/profiles.json
"""

import argparse
import glob
import json

import numpy as np

# (glob, sigma_min, sigma_max, t_clip) per cell. t_clip must match what the run
# was trained with -- src.infonoise divides the loss weight back out, so a wrong
# value here silently changes the reconstructed density.
ARMS = {
    # The six cells as currently run: the corrected estimator grid, where
    # sigma_max is 80 for every scale instead of scaling with k (commit 3983f62),
    # and t_clip scaled with the data on the k=0.1 arms.
    "mnist": ("ds-mnist__cond-none__dist-infonoise__seed-*", 0.002, 80.0, 0.05),
    "mnist_x10": (
        "ds-mnist_x10__cond-none__dist-infonoise_sigma_min_0.02_sigma_max_80.0__seed-*",
        0.02,
        80.0,
        0.05,
    ),
    "mnist_x0.1": (
        "ds-mnist_x0.1__cond-none__dist-infonoise_sigma_min_0.0002_sigma_max_80.0_tclip_0.005__seed-*",
        0.0002,
        80.0,
        0.005,
    ),
    "cifar10": ("ds-cifar10__cond-none__dist-infonoise__seed-*", 0.002, 80.0, 0.05),
    "cifar10_x10": (
        "ds-cifar10_x10__cond-none__dist-infonoise_sigma_min_0.02_sigma_max_80.0__seed-*",
        0.02,
        80.0,
        0.05,
    ),
    "cifar10_x0.1": (
        "ds-cifar10_x0.1__cond-none__dist-infonoise_sigma_min_0.0002_sigma_max_80.0_tclip_0.005__seed-*",
        0.0002,
        80.0,
        0.005,
    ),
    # The superseded grid (sigma_max = 80k), kept so the two can be plotted
    # against each other -- this is the arm whose FID is already scored.
    "mnist_x10_oldgrid": (
        "ds-mnist_x10__cond-none__dist-infonoise_sigma_min_0.02_sigma_max_800.0__seed-*",
        0.02,
        800.0,
        0.05,
    ),
    "mnist_x0.1_oldgrid": (
        "ds-mnist_x0.1__cond-none__dist-infonoise_sigma_min_0.0002_sigma_max_8.0_tclip_0.005__seed-*",
        0.0002,
        8.0,
        0.005,
    ),
    "cifar10_x10_oldgrid": (
        "ds-cifar10_x10__cond-none__dist-infonoise_sigma_min_0.02_sigma_max_800.0__seed-*",
        0.02,
        800.0,
        0.05,
    ),
    "cifar10_x0.1_oldgrid": (
        "ds-cifar10_x0.1__cond-none__dist-infonoise_sigma_min_0.0002_sigma_max_8.0_tclip_0.005__seed-*",
        0.0002,
        8.0,
        0.005,
    ),
}

# best swept training centre per cell, in sigma, from the centre sweeps
TUNED = {
    "mnist": 1.0,
    "mnist_x10": 10.0,
    "mnist_x0.1": 0.2,
    "cifar10": 0.5,
    "cifar10_x10": 2.0,
    "cifar10_x0.1": 0.1,
}
TUNED.update({f"{k}_oldgrid": v for k, v in list(TUNED.items())})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/infonoise_profiles.json")
    p.add_argument("--logs", default="logs/metrics")
    args = p.parse_args()

    out = {}
    for cell, (pattern, smin, smax, t_clip) in ARMS.items():
        m_hats, pis, pis_first, gates = [], [], [], []
        centers = None
        for path in sorted(glob.glob(f"{args.logs}/{pattern}/infonoise_profile.jsonl")):
            lines = [ln for ln in open(path).read().strip().split("\n") if ln]
            if not lines:
                continue
            last = json.loads(lines[-1])
            centers = np.array(last["sigma_centers"])
            m_hats.append(np.array(last["m_hat"]))
            pis.append(np.array(last["pi"]))
            gates.append(last["gate_c"])
            pis_first.append(np.array(json.loads(lines[0])["pi"]))
        if not pis:
            print(f"  (no profiles for {cell})")
            continue

        pi = np.median(pis, axis=0)
        d_log = float(np.log(centers[1]) - np.log(centers[0]))
        cdf = np.cumsum(pi) * d_log
        cdf /= cdf[-1]
        out[cell] = {
            "sigma": centers.tolist(),
            "m_hat": np.median(m_hats, axis=0).tolist(),
            "pi": pi.tolist(),
            "pi_first": np.median(pis_first, axis=0).tolist(),
            "gate_c": float(np.median(gates)),
            "sigma_min": smin,
            "sigma_max": smax,
            "t_clip": t_clip,
            "tuned_sigma": TUNED[cell],
            "n_seeds": len(pis),
            "median": float(np.interp(0.5, cdf, centers)),
            "q10": float(np.interp(0.1, cdf, centers)),
            "q90": float(np.interp(0.9, cdf, centers)),
        }
        print(f"  {cell:<14} {len(pis)} seeds, gate_c={out[cell]['gate_c']:.4g}")

    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out} ({len(out)} cells)")


if __name__ == "__main__":
    main()
