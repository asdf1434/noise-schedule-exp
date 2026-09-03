"""Rescore a few July-era baseline folders with the CURRENT (August) FID path,
and with the OLD (July) folder-vs-folder path, and compare both to the value
stored in results/master_fid_results.json.

The stored baseline numbers were computed in July and never recomputed, while
the InfoNoise arms were scored in August with a rewritten evaluate_fid.py. If
the two paths disagree, the InfoNoise-vs-baseline comparison is confounded.
"""
import json, os, sys

# `python scripts/fairness_check.py` puts scripts/ on sys.path, not the repo
# root -- without this, `import src` fails (job 1655630 died exactly here)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from cleanfid import fid
from src.datasets import DATASETS
from src.fid_weights import ensure_inception_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec = DATASETS["mnist"]
REAL_DIR, STATS = spec.real_dir, spec.real_stats_name
print("device:", DEVICE, "| real_dir:", REAL_DIR, "| stats:", STATS, flush=True)
print("real images on disk:", len(os.listdir(REAL_DIR)), flush=True)

ensure_inception_weights()
feat_model = fid.build_feature_extractor("clean", DEVICE)
ref_mu, ref_sigma = fid.get_reference_statistics(STATS, res="na", mode="clean", split="custom")
print("cached ref stats loaded", flush=True)

master = json.load(open("results/master_fid_results.json"))

def august(d):
    f = fid.get_folder_features(d, model=feat_model, num_workers=16, batch_size=128,
                                device=DEVICE, mode="clean", verbose=False)
    return float(fid.frechet_distance(np.mean(f, 0), np.cov(f, rowvar=False), ref_mu, ref_sigma))

def july(d):
    return float(fid.compute_fid(REAL_DIR, d, mode="clean", device=DEVICE,
                                 num_workers=16, batch_size=128, verbose=False))

TARGETS = []
for s in (0, 1, 2):
    for sch in ("uniform", "shifted_coarse", "logit_normal", "shifted_fine"):
        TARGETS.append((f"logit_normal_mu_0.0_sigma_1.0_seed{s}", "epoch_100", sch))
# one InfoNoise folder as a control: it was scored in August, so both paths
# should also agree there
TARGETS.append(("ds-mnist__cond-none__dist-infonoise__seed-0", "epoch_100", "uniform"))

print(f"\n{'experiment':52s} {'sched':15s} {'stored':>8s} {'august':>8s} {'july':>8s} {'d_aug':>7s}")
rows = []
for exp, ep, sch in TARGETS:
    d = os.path.join("eval_runs", exp, ep, sch)
    if not os.path.isdir(d) or not os.listdir(d):
        print(f"{exp:52s} {sch:15s}  MISSING DIR {d}"); continue
    stored = master.get(exp, {}).get(sch, {}).get(ep)
    a, j = august(d), july(d)
    rows.append((exp, sch, stored, a, j))
    ds = f"{a-stored:+.3f}" if stored is not None else "n/a"
    print(f"{exp:52s} {sch:15s} {stored if stored is not None else float('nan'):8.3f} {a:8.3f} {j:8.3f} {ds:>7s}", flush=True)

base = [r for r in rows if r[0].startswith("logit_normal") and r[2] is not None]
if base:
    d = [r[3]-r[2] for r in base]
    print(f"\nbaseline rescore drift (august path - stored july value): "
          f"mean {np.mean(d):+.4f}, max |d| {max(abs(x) for x in d):.4f}, n={len(d)}")
    d2 = [r[3]-r[4] for r in rows]
    print(f"august path - july path on same folders: mean {np.mean(d2):+.4f}, "
          f"max |d| {max(abs(x) for x in d2):.4f}, n={len(d2)}")
    print("\nVERDICT: comparison is FAIR on the FID axis" if max(abs(x) for x in d) < 0.5
          else "\nVERDICT: FID paths DISAGREE -- stored baselines are not comparable")
