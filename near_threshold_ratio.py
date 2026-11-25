#!/usr/bin/env python3
"""Behaviour of b_trans/b_crit as R0 -> 1+ with a fixed initial seed.

b_crit -> 0 linearly in (R0 - 1), while the seed alone drives P to O(i(0)),
so the ratio passes through 1 and then diverges as R0 -> 1.
Output: results/near_threshold_ratio.json
"""
import json
from pathlib import Path
import numpy as np
from verify_revision import P0, R0_value
from thresholds import transient_threshold_certified, check_threshold_ordering

OUT = Path(__file__).resolve().parent / "results"
B_INV = (P0[2] + P0[1]) / P0[2] / (1.0 + P0[11] * P0[3] / (P0[4] + P0[6]))
rows = []
for seed in [1e-6, 1e-8]:
    for eps in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        p = P0.copy(); p[0] = B_INV * (1 + eps)
        y0 = np.zeros(6); y0[2] = seed; y0[0] = 1 - seed
        try:
            r = transient_threshold_certified(p, y0=y0, T_max=4.0e6)
            check_threshold_ordering(r)
            rows.append(dict(seed=seed, R0_minus_1=float(R0_value(p) - 1), bcrit=r.bcrit, btrans=r.btrans,
                             ratio=r.ratio, tau_star=r.tau_star, horizon=r.horizon,
                             P_star_over_seed=r.btrans / seed))
            print(f"seed={seed:g} R0-1={R0_value(p)-1:.2e} bcrit={r.bcrit:.3e} btrans={r.btrans:.3e} "
                  f"ratio={r.ratio:.3f} tau*={r.tau_star:.0f} T={r.horizon:.0f}", flush=True)
        except Exception as exc:
            print(f"seed={seed:g} eps={eps:g} FAILED {exc}", flush=True)
            rows.append(dict(seed=seed, eps=eps, error=str(exc)))
json.dump(rows, open(OUT / "near_threshold_ratio.json", "w"), indent=1)
