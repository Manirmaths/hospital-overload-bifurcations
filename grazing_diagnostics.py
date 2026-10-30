#!/usr/bin/env python3
"""R2.5: numerical convergence diagnostics for Theorem 7.2 (grazing laws).

The full piecewise-smooth system is integrated *phase by phase* with event
location on Psi = 0 (normal care -> capacity-limited -> normal care), so the
switching times are resolved to integrator tolerance rather than by sampling.
For eps = b_trans - b on a geometric grid and for two tolerance levels we report
  * Delta tau / eps^{1/2}          (-> 2 sqrt(2/k))
  * D_adm / eps^{3/2}               (-> 4 sqrt(2) c / (3 sqrt(k)))
  * entry-time shift tau_1 - tau_1^N (should be exactly 0: trajectories coincide)
  * exit-time shift tau_2 - tau_2^N (predicted O(eps))
  * sup_{0<=tau<=T} ||y - y_N||_inf / eps^{3/2} for T = 200 and T = 2000
Output: results/grazing_diagnostics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

from verify_revision import P0, Y0, rhs_normal
from thresholds import transient_threshold_certified

OUT = Path(__file__).resolve().parent / "results"
B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, P0)


def f_cap(t, y, b):
    s, e, i, q, h, r = y
    n = y.sum()
    inc = B * s * (i + theta * q) / n
    A = c * (b - h)
    return np.array([m * (1 - s) - inc, inc - (eta + m) * e, eta * e - i,
                     pi * i - A - chiQ * q, A - chiH * h, rhoI * i + rhoQ * q + rhoH * h - m * r])


def run_full(b, T, rtol, atol, ts=None, dtau=None):
    """Phase-by-phase integration with event location on Psi=0. Inside the
    window [ts-1, ts+1] the step is capped at dtau/20 so that very short
    capacity-limited episodes cannot be stepped over."""
    def psi_ev(t, y, *args):
        return a * y[3] - c * (b - y[4])
    psi_ev.terminal = True
    breaks = sorted(set([0.0, T] + ([max(ts - 1, 0), ts + 1] if ts is not None else [])))
    segs, switch_times = [], []
    t0, y0, regime = 0.0, Y0.copy(), "N"
    while t0 < T - 1e-12:
        nxt = min(x for x in breaks if x > t0 + 1e-12)
        in_win = ts is not None and ts - 1 <= t0 < ts + 1
        ms = max(dtau / 20, 1e-7) if in_win else np.inf
        f = (lambda t, y: rhs_normal(t, y)) if regime == "N" else (lambda t, y: f_cap(t, y, b))
        psi_ev.direction = 1 if regime == "N" else -1
        sol = solve_ivp(f, (t0, nxt), y0, method="DOP853", rtol=rtol, atol=atol,
                        events=psi_ev, dense_output=True, max_step=ms)
        segs.append((regime, sol))
        if sol.status == 1 and sol.t_events[0][0] > t0 + 1e-13:
            t0 = float(sol.t_events[0][0]); y0 = sol.y_events[0][0].copy()
            switch_times.append(t0)
            regime = "C" if regime == "N" else "N"
            continue
        t0, y0 = float(sol.t[-1]), sol.y[:, -1].copy()
    return segs, switch_times


def evaluate(segs, t):
    for reg, sol in segs:
        if sol.t[0] <= t <= sol.t[-1]:
            return sol.sol(t)
    return segs[-1][1].sol(t)


def main():
    tr = transient_threshold_certified(P0)
    bt, ts = tr.btrans, tr.tau_star
    ref = solve_ivp(lambda t, y: rhs_normal(t, y), (0, 2000), Y0, method="DOP853",
                    rtol=1e-13, atol=1e-16, dense_output=True)
    Pfun = lambda t: ref.sol(t)[4] + a / c * ref.sol(t)[3]
    dP = lambda t: (lambda y, dy: dy[4] + a / c * dy[3])(ref.sol(t), rhs_normal(t, ref.sol(t)))
    hh = 1e-4
    k = -(dP(ts + hh) - dP(ts - hh)) / (2 * hh)
    Kdur = 2 * np.sqrt(2 / k)
    Kdef = 4 * np.sqrt(2) * c / (3 * np.sqrt(k))
    out = dict(btrans=bt, tau_star=ts, k=k, duration_constant=Kdur, deficit_constant=Kdef, runs={})
    eps_grid = np.logspace(-9, -3, 13) * bt / 5.488e-2
    for rtol, atol in [(1e-10, 1e-14), (1e-12, 1e-16)]:
        rows = []
        for eps in eps_grid:
            b = bt - eps
            segs, sw = run_full(b, 2000.0, rtol, atol, ts=ts, dtau=2 * np.sqrt(2 * eps / k))
            t1, t2 = sw[0], sw[1]
            # reference (normal-care) crossing times
            t1N = brentq(lambda t: Pfun(t) - b, ts - 3 * np.sqrt(2 * eps / k) - 1e-3, ts, xtol=1e-14)
            t2N = brentq(lambda t: Pfun(t) - b, ts, ts + 3 * np.sqrt(2 * eps / k) + 1e-3, xtol=1e-14)
            capsol = [s for r, s in segs if r == "C"][0]
            D = quad(lambda t: a * capsol.sol(t)[3] - c * (b - capsol.sol(t)[4]), t1, t2,
                     epsabs=1e-22, epsrel=1e-12, limit=400)[0]
            dev = {}
            for T in (200.0, 2000.0):
                tt = np.linspace(0, T, 40001)
                Yf = np.array([evaluate(segs, t) for t in tt]).T
                Yn = ref.sol(tt)
                dev[str(int(T))] = float(np.max(np.abs(Yf - Yn)))
            rows.append(dict(eps=float(eps), n_switches=len(sw), dur=t2 - t1, deficit=D,
                             dur_comp=(t2 - t1) / np.sqrt(eps), def_comp=D / eps ** 1.5,
                             entry_shift=t1 - t1N, exit_shift=t2 - t2N,
                             exit_shift_over_eps=(t2 - t2N) / eps,
                             dev200_comp=dev["200"] / eps ** 1.5, dev2000_comp=dev["2000"] / eps ** 1.5))
            print(f"rtol={rtol:g} eps={eps:.2e} nsw={len(sw)} dur/K={rows[-1]['dur_comp']/Kdur:.6f} "
                  f"def/K={rows[-1]['def_comp']/Kdef:.6f} entry={t1-t1N:.1e} exit/eps={(t2-t2N)/eps:.3e} "
                  f"dev200={rows[-1]['dev200_comp']:.4e} dev2000={rows[-1]['dev2000_comp']:.4e}", flush=True)
        out["runs"][f"{rtol:g}"] = rows
    json.dump(out, open(OUT / "grazing_diagnostics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
