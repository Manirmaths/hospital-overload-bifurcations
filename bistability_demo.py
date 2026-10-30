#!/usr/bin/env python3
"""R2.3: direct demonstration of bistability at b = 1.5e-4 (in (b_LPC^-, b_H))
and b = 1.5e-3 (in (b_crit, b_LPC^+)).

For each b:
  * trajectories from a small perturbation of the stable equilibrium and from a
    point on the large attracting cycle (computed at b itself);
  * a one-parameter family y(lam) = E + lam (y_cyc - E) to locate the basin
    boundary along that segment;
  * a 2-D basin slice in (q, h) with (s, e, i, r) fixed at the equilibrium.
Integration: LSODA, rtol 1e-9, atol 1e-12, t_end = 8000 (about 35 periods).
Classification: peak-to-peak range of q over the final 1000 time units.
Output: results/bistability.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp

from verify_revision import P0, normal_equilibrium, capacity_equilibria

OUT = Path(__file__).resolve().parent / "results"
B_, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, P0)
_, BCRIT = normal_equilibrium()
yN, _ = normal_equilibrium()
T_END = 8000.0


def rhs(t, y, b):
    s, e, i, q, h, r = y
    n = s + e + i + q + h + r
    inc = B_ * s * (i + theta * q) / n
    A = min(a * q, c * max(b - h, 0.0))
    return [m * (1 - s) - inc, inc - (eta + m) * e, eta * e - i,
            pi * i - A - chiQ * q, A - chiH * h, rhoI * i + rhoQ * q + rhoH * h - m * r]


def run(args):
    y0, b, t_end = args
    sol = solve_ivp(rhs, (0, t_end), y0, args=(b,), method="LSODA", rtol=1e-9, atol=1e-12,
                    dense_output=True)
    tt = np.linspace(t_end - 1000, t_end, 2001)
    yy = sol.sol(tt)
    amp = float(np.ptp(yy[3]))
    return amp


def equilibrium(b):
    if b >= BCRIT:
        return yN.copy()
    return capacity_equilibria(b)[0]


def cycle_point(b):
    """Point on the attracting cycle at capacity b (by continuation in b from 0.5 bcrit)."""
    y = None
    b_seed = 0.5 * BCRIT
    eq = capacity_equilibria(b_seed)[0]
    y = eq.copy(); y[2] *= 1.0 + 1e-3
    path = np.geomspace(b_seed, b, 6)
    for bb in path:
        sol = solve_ivp(rhs, (0, 6000), y, args=(bb,), method="LSODA", rtol=1e-10, atol=1e-13,
                        dense_output=True)
        tt = np.linspace(5000, 6000, 20001)
        yy = sol.sol(tt)
        j = int(np.argmax(yy[3]))
        y = yy[:, j]
    return y


def trajectory(y0, b, t_end=T_END, npts=4001):
    sol = solve_ivp(rhs, (0, t_end), y0, args=(b,), method="LSODA", rtol=1e-9, atol=1e-12,
                    dense_output=True)
    tt = np.linspace(0, t_end, npts)
    return tt, sol.sol(tt)


def main(n_grid=21, workers=2):
    out = {}
    for b in [1.5e-4, 1.5e-3]:
        E = equilibrium(b)
        yc = cycle_point(b)
        rec = dict(b=b, b_over_bcrit=b / BCRIT, equilibrium=E.tolist(), cycle_point=yc.tolist())
        # (i) two trajectories
        ye = E.copy(); ye[2] *= 1.01; ye[3] *= 1.01
        tt, Ye = trajectory(ye, b)
        tt, Yc = trajectory(yc, b)
        rec["traj_t"] = tt[::4].tolist()
        rec["traj_eq_q"] = Ye[3, ::4].tolist(); rec["traj_eq_h"] = Ye[4, ::4].tolist()
        rec["traj_cyc_q"] = Yc[3, ::4].tolist(); rec["traj_cyc_h"] = Yc[4, ::4].tolist()
        rec["final_amp_eq_start"] = float(np.ptp(Ye[3, -500:]))
        rec["final_amp_cycle_start"] = float(np.ptp(Yc[3, -500:]))
        # (ii) segment family
        lams = np.linspace(0, 1.0, 41)
        jobs = [(E + lam * (yc - E), b, T_END) for lam in lams]
        with ProcessPoolExecutor(workers) as ex:
            amps = list(ex.map(run, jobs))
        rec["segment_lambda"] = lams.tolist(); rec["segment_amp"] = amps
        # (iii) 2-D slice in (q, h)
        qg = np.linspace(0.0, 1.25 * max(yc[3], 0.013), n_grid)
        hg = np.linspace(0.0, 0.999 * b, n_grid)
        jobs = []
        for hv in hg:
            for qv in qg:
                y0 = E.copy(); y0[3] = max(qv, 1e-12); y0[4] = hv
                jobs.append((y0, b, T_END))
        with ProcessPoolExecutor(workers) as ex:
            amps2 = list(ex.map(run, jobs, chunksize=4))
        rec["slice_q"] = qg.tolist(); rec["slice_h"] = hg.tolist()
        rec["slice_amp"] = np.array(amps2).reshape(n_grid, n_grid).tolist()
        out[f"{b:g}"] = rec
        print(b, "eq-start amp", rec["final_amp_eq_start"], "cycle-start amp", rec["final_amp_cycle_start"],
              "segment", np.round(amps, 5).tolist(), flush=True)
    json.dump(out, open(OUT / "bistability.json", "w"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    main(n_grid=n)
