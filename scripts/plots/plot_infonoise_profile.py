"""Plot what InfoNoise learned: the online information-profile estimate and the
training noise distribution it induces, over the course of a run.

Reads logs/metrics/<exp_name>/infonoise_profile.jsonl (one record per sampler
refresh, written by src/infonoise.py) and reproduces the paper's Figure 4
decomposition, plus how the sampler moved away from the warm-up prior:

  (a) m_hat(sigma)         -- the binned unweighted denoising loss
  (b) rho_hat(sigma)       -- the gated entropy-rate profile, the allocation target
  (c) pi(sigma)            -- the sampling density, rho_hat/w
  (d) pi in t coordinates  -- directly comparable to plot_train_dists.py

Usage:
    python -m scripts.plots.plot_infonoise_profile --experiment ds-mnist__cond-none__dist-infonoise__seed-42
    python -m scripts.plots.plot_infonoise_profile --experiment <name> --save_path plots/infonoise.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_records(experiment: str) -> list[dict]:
    path = os.path.join("logs", "metrics", experiment, "infonoise_profile.jsonl")
    if not os.path.exists(path):
        raise SystemExit(
            f"No InfoNoise profile log at {path} -- was this run trained with "
            "--train_dist infonoise?"
        )
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit(f"{path} is empty (no sampler refresh happened)")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--max_curves",
        type=int,
        default=8,
        help="refreshes to draw, evenly spaced through the run (earliest and "
        "latest always included)",
    )
    args = parser.parse_args()

    records = load_records(args.experiment)
    picks = np.unique(
        np.linspace(0, len(records) - 1, min(args.max_curves, len(records))).astype(int)
    )
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    panels = [
        ("m_hat", r"$\hat{m}(\sigma)$  (unweighted denoising loss)", True),
        ("rho_hat", r"$\hat{\rho}(\sigma)$  (gated entropy-rate profile)", False),
        ("pi", r"$\pi(\sigma) \propto \hat{\rho}/w$  (sampling density)", False),
    ]

    for ax, (key, title, logy) in zip(axes, panels):
        for i, idx in enumerate(picks):
            rec = records[idx]
            color = cmap(i / max(len(picks) - 1, 1))
            ax.plot(
                rec["sigma_centers"],
                rec[key],
                color=color,
                lw=1.6,
                label=f"step {rec['step']}",
            )
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"noise scale $\sigma = (1-t)/t$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    # gate pivot of the final refresh, the one knob that decides how much of the
    # low-noise end gets suppressed
    gate_c = records[-1]["gate_c"]
    if gate_c:
        for ax in axes[:3]:
            ax.axvline(
                gate_c, color="crimson", ls="--", lw=1.0, label=f"gate c={gate_c:.3g}"
            )

    # pi re-expressed in t, so it can be laid next to plot_train_dists.py output
    ax = axes[3]
    for i, idx in enumerate(picks):
        rec = records[idx]
        sigma = np.asarray(rec["sigma_centers"])
        pi_log_sigma = np.asarray(rec["pi"])
        t = 1.0 / (1.0 + sigma)
        # density transform: pi(t) = pi(log sigma) * |d log sigma / dt|,
        # and d(log sigma)/dt = -1 / (t (1 - t))
        pi_t = pi_log_sigma / (t * (1.0 - t))
        order = np.argsort(t)
        ax.plot(
            t[order],
            pi_t[order],
            color=cmap(i / max(len(picks) - 1, 1)),
            lw=1.6,
            label=f"step {rec['step']}",
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("t  (0=noise, 1=clean)")
    ax.set_title(r"$\pi(t)$  (training noise distribution)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"InfoNoise online adaptation -- {args.experiment}", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_path = args.save_path or os.path.join(
        "plots", f"infonoise_profile_{args.experiment}.png"
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
