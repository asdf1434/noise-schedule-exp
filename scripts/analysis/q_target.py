"""The q_hat target of Section 3.1 (Eq. 13) of arXiv:2602.18647, with the loss
estimate m_hat replaced by the exact B.4 mmse from results/mmse_b4/.

    per sigma:  q(sigma)  ∝ mmse(sigma) / sigma^3          Eq. (13), (59)
    per lambda: q(lambda) ∝ (1/2) mmse(sigma) / sigma^2     Eq. (61), lambda = -2 log sigma

Ungated. Each is normalized to integrate to 1 over the 61-point grid in its own
coordinate (trapezoid rule). Runs on every file in results/mmse_b4/, including
the per-class ones.

    python scripts/analysis/q_target.py
"""

import glob
import json
import os

import numpy as np

os.makedirs("results/q_target", exist_ok=True)
for path in sorted(glob.glob("results/mmse_b4/*.json")):
    name = os.path.basename(path)[: -len(".json")]
    with open(path) as f:
        r = json.load(f)
    sigma = np.array(r["sigma"])
    mmse = np.array(r["mmse"])
    lam = -2 * np.log(sigma)

    q_sigma = mmse / sigma**3
    q_sigma /= np.trapezoid(q_sigma, sigma)

    q_lambda = 0.5 * mmse / sigma**2
    # lam decreases along the grid, so integrate in reverse
    q_lambda /= np.trapezoid(q_lambda[::-1], lam[::-1])

    with open(f"results/q_target/{name}.json", "w") as f:
        json.dump(
            {
                "dataset": name,
                "sigma": sigma.tolist(),
                "lambda": lam.tolist(),
                "q_sigma": q_sigma.tolist(),
                "q_lambda": q_lambda.tolist(),
            },
            f,
            indent=1,
        )
    print(
        f"{name:14s} peak per sigma at sigma={sigma[q_sigma.argmax()]:.3g}, "
        f"peak per lambda at lambda={lam[q_lambda.argmax()]:.2f} (sigma={sigma[q_lambda.argmax()]:.3g})"
    )
