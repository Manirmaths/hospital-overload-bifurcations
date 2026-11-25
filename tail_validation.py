#!/usr/bin/env python3
"""Numerical convergence evidence for the stopping rule of thresholds.py.

The tail bound used to stop the b_trans search comes from the linearisation at
E_N, with a heuristic allowance for the nonlinear remainder; it is therefore a
numerical stopping criterion, not a rigorous proof. For every atlas value of B
this script checks, independently of the stopping rule:
  1. horizon extension: re-integrate to T_ext = max(8 T_stop, 2e4) at a tighter
     tolerance (rtol 1e-12, atol 1e-15) and recompute the global maximum of P;
  2. bound validity: compare the actual sup_{T_stop<=tau<=T_ext} |P - b_crit|
     with the (inflated) linearised bound used at T_stop.
Output: results/tail_validation.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from verify_revision import P0, Y0, rhs_normal
from thresholds import transient_threshold_certified

OUT = Path(__file__).resolve().parent / "results"


def global_max(p, T, rtol, atol):
    a, c = p[4], p[5]
    sol = solve_ivp(lambda t, y: rhs_normal(t, y, p), (0, T), Y0, method="DOP853", rtol=rtol, atol=atol,
                    dense_output=True, max_step=max(1.0, T / 2.0e4))
    dt = max(0.02, T / 4.0e5)
    ts = np.arange(0, T + 0.5 * dt, dt)
    Y = sol.sol(ts)
    P = Y[4] + a / c * Y[3]
    idx = np.where((P[1:-1] >= P[:-2]) & (P[1:-1] > P[2:]))[0] + 1
    best = (P[0], 0.0)
    for j in idx:
        o = minimize_scalar(lambda t: -(sol.sol(t)[4] + a / c * sol.sol(t)[3]),
                            bounds=(ts[j - 1], ts[j + 1]), method="bounded", options={"xatol": 1e-11})
        if -o.fun > best[0]:
            best = (-o.fun, o.x)
    return best, ts, P


def main():
    atlas = json.load(open(OUT / "threshold_atlas.json"))["rows"]
    rows = []
    for a_row in atlas:
        p = P0.copy(); p[0] = a_row["B"]
        r = transient_threshold_certified(p)
        T_ext = max(8.0 * r.horizon, 2.0e4)
        (bt_ext, ts_ext), ts, P = global_max(p, T_ext, 1e-12, 1e-15)
        tail = ts >= r.horizon
        actual_tail = float(np.max(np.abs(P[tail] - r.bcrit)))
        rows.append(dict(B=a_row["B"], R0=a_row["R0"], bcrit=r.bcrit, T_stop=r.horizon, T_ext=T_ext,
                         btrans_stop=r.btrans, btrans_ext=float(bt_ext),
                         rel_change=float((bt_ext - r.btrans) / r.btrans),
                         tau_star_stop=r.tau_star, tau_star_ext=float(ts_ext),
                         tail_bound=r.tail_bound, actual_tail_sup=actual_tail,
                         bound_holds=bool(actual_tail <= r.tail_bound),
                         bound_ratio=float(actual_tail / r.tail_bound),
                         tail_over_gap=float(actual_tail / (bt_ext - r.bcrit))))
        x = rows[-1]
        print(f"B={x['B']:.4f} Tstop={x['T_stop']:.0f} rel={x['rel_change']:.1e} "
              f"tail/bound={x['bound_ratio']:.3f} tail/gap={x['tail_over_gap']:.3f}", flush=True)
    json.dump(rows, open(OUT / "tail_validation.json", "w"), indent=1)
    print("max |rel change|", max(abs(x["rel_change"]) for x in rows),
          "| bound exceeded at", sum(not x["bound_holds"] for x in rows), "of", len(rows),
          "| max tail/bound", max(x["bound_ratio"] for x in rows),
          "| max tail/gap", max(x["tail_over_gap"] for x in rows))


if __name__ == "__main__":
    main()
