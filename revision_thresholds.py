#!/usr/bin/env python3
"""R2.1 / R2.4: regenerate the threshold atlas with the certified b_trans search
and test initial-condition dependence of b_trans/b_crit.

Outputs: results/threshold_atlas.json, results/ic_robustness.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from numpy.linalg import eigvals

from verify_revision import P0, R0_value, normal_equilibrium, capacity_equilibria, jacobian
from thresholds import transient_threshold_certified, check_threshold_ordering

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

B_INV = (P0[2] + P0[1]) / P0[2] / (1.0 + P0[11] * P0[3] / (P0[4] + P0[6]))


def hopf_b(p):
    _, bc = normal_equilibrium(p)

    def alphaC(b):
        roots = capacity_equilibria(b, p)
        if not roots:
            return np.nan
        return float(np.max(eigvals(jacobian(roots[0], b, "C", p)).real))

    grid = np.geomspace(bc * 1e-4, bc * (1 - 1e-8), 120)
    vals = np.array([alphaC(x) for x in grid])
    idx = np.where(vals[:-1] * vals[1:] < 0)[0]
    out = [brentq(alphaC, grid[k], grid[k + 1], xtol=1e-15) for k in idx]
    return out


def atlas():
    Bvals = np.unique(np.r_[B_INV * np.array([1.005, 1.01, 1.02, 1.04, 1.07, 1.1]),
                            np.linspace(1.08, 6.0, 83), 1.13889])
    rows = []
    old_rows = []
    from verify_revision import transient_threshold_v100 as old_tt
    for B in Bvals:
        p = P0.copy(); p[0] = B
        R0 = R0_value(p)
        if R0 <= 1:
            continue
        try:
            r = transient_threshold_certified(p)
        except RuntimeError as exc:
            print('B', B, 'not certified:', exc, flush=True)
            continue
        check_threshold_ordering(r)
        hb = hopf_b(p)
        try:
            bt_old = float(old_tt(p, 160.0)[0])
        except Exception:
            bt_old = float("nan")
        rows.append(dict(B=float(B), R0=float(R0), bcrit=r.bcrit, btrans=r.btrans,
                         tau_star=r.tau_star, horizon=r.horizon, n_local_maxima=r.n_local_maxima,
                         tail_bound=r.tail_bound, ratio=r.ratio, bH=hb,
                         btrans_v100=bt_old,
                         v100_violates_theorem=bool(bt_old < r.bcrit),
                         v100_rel_error=float((bt_old - r.btrans) / r.btrans),
                         secondary_maxima=[m for m in r.local_maxima[1:4]]))
        print(f"B={B:.4f} R0={R0:.4f} bcrit={r.bcrit:.4e} btrans={r.btrans:.4e} tau*={r.tau_star:.1f} "
              f"old={bt_old:.4e} bH={hb}", flush=True)
    json.dump(dict(B_invasion=float(B_INV), rows=rows), open(OUT / "threshold_atlas.json", "w"), indent=1)
    return rows


def ic_cases():
    """Epidemiologically plausible alternative initial conditions."""
    cases = {}
    for i0 in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        y = np.zeros(6); y[2] = i0; y[0] = 1 - i0
        cases[f"i0={i0:g}"] = y
    for e0 in [1e-6, 1e-4]:
        y = np.zeros(6); y[1] = e0; y[0] = 1 - e0
        cases[f"e0={e0:g}"] = y
    # 10% and 30% prior immunity, seed i0 = 1e-6
    for imm in [0.1, 0.3]:
        y = np.zeros(6); y[2] = 1e-6; y[5] = imm; y[0] = 1 - imm - 1e-6
        cases[f"immune={imm:g},i0=1e-6"] = y
    return cases


def ic_robustness():
    out = {}
    Bset = [1.2, 1.5, 2.0, 3.0, 4.0, 6.0]
    for name, y0 in ic_cases().items():
        out[name] = []
        for B in Bset:
            p = P0.copy(); p[0] = B
            # with prior immunity R0 is unchanged, but the effective reproduction
            # number may be < 1; record and skip non-outbreak cases
            Reff = R0_value(p) * y0[0]
            r = transient_threshold_certified(p, y0=y0)
            check_threshold_ordering(r)
            out[name].append(dict(B=B, Reff0=float(Reff), btrans=r.btrans, bcrit=r.bcrit,
                                  ratio=r.ratio, tau_star=r.tau_star))
        print(name, [f"{x['ratio']:.2f}" for x in out[name]], flush=True)
    json.dump(out, open(OUT / "ic_robustness.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    atlas()
    ic_robustness()
