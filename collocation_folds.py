#!/usr/bin/env python3
"""R2.2 / R1.1: independent collocation + pseudo-arclength continuation of the
periodic-orbit branch and its two folds.

Independence from the v1.0.0 shooting workflow
----------------------------------------------
* discretisation: piecewise-cubic Lobatto-IIIA collocation with adaptive mesh
  (scipy.integrate.solve_bvp) instead of single shooting with DOP853;
* coordinates: logarithmic state u = log y (all components stay positive),
  which equilibrates the six orders of magnitude spanned by q along the cycle;
* continuation: pseudo-arclength in (u(0), T, b) with a secant predictor,
  so folds are passed as regular turning points rather than approached in a
  monotone parameter;
* fold detection: geometric (turning point of b along arclength, located by
  local quadratic interpolation); no Floquet multipliers or variational
  equations are used here, so the result does not depend on how the
  monodromy matrix treats crossings of the switching surface.

Usage:  python collocation_folds.py [tol]
Output: results/collocation_branch_tol{tol}.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp

from verify_revision import P0, normal_equilibrium, simulate_cycle

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, P0)
_, BCRIT = normal_equilibrium()
BS = 1e-3     # scale for b in the arclength norm
TS = 100.0    # scale for T in the arclength norm


def f_vec(Y, b):
    s, e, i, q, h, r = Y
    n = Y.sum(axis=0)
    inc = B * s * (i + theta * q) / n
    A = np.minimum(a * q, c * np.maximum(b - h, 0.0))
    return np.array([m * (1 - s) - inc, inc - (eta + m) * e, eta * e - i,
                     pi * i - A - chiQ * q, A - chiH * h,
                     rhoI * i + rhoQ * q + rhoH * h - m * r])


def g_log(U, b):
    Y = np.exp(U)
    return f_vec(Y, b) / Y


class Branch:
    def __init__(self, tol=1e-8, max_nodes=200000):
        self.tol = tol
        self.max_nodes = max_nodes

    def solve(self, sgrid, U, T, b, z_prev=None, tangent=None, ds=None, fixed_b=False):
        if fixed_b:
            def fun(s, Uv, p):
                return p[0] * g_log(Uv, b)

            def bc(ua, ub, p):
                return np.r_[ua - ub, g_log(ua[:, None], b)[3, 0]]
            sol = solve_bvp(fun, bc, sgrid, U, p=np.array([T]), tol=self.tol,
                            max_nodes=self.max_nodes, bc_tol=1e-10)
            return sol, sol.p[0], b

        def fun(s, Uv, p):
            return p[0] * g_log(Uv, p[1])

        def bc(ua, ub, p):
            z = np.r_[ua, p[0] / TS, p[1] / BS]
            return np.r_[ua - ub, g_log(ua[:, None], p[1])[3, 0], (z - z_prev) @ tangent - ds]
        sol = solve_bvp(fun, bc, sgrid, U, p=np.array([T, b]), tol=self.tol,
                        max_nodes=self.max_nodes, bc_tol=1e-10)
        return sol, sol.p[0], sol.p[1]


def orbit_summary(sol, T, b):
    s = np.linspace(0, 1, 8001)
    Y = np.exp(sol.sol(s))
    psi = a * Y[3] - c * (b - Y[4])
    # transversality of each crossing: |dPsi/dtau| at the crossings
    dpsi = a * f_vec(Y, b)[3] + c * f_vec(Y, b)[4]
    idx = np.where(psi[:-1] * psi[1:] < 0)[0]
    return dict(b=float(b), T=float(T), qmin=float(Y[3].min()), qmax=float(Y[3].max()),
                ampq=float(np.ptp(Y[3])), hmax=float(Y[4].max()),
                psimin=float(psi.min()), psimax=float(psi.max()),
                fracC=float(np.mean(psi >= 0)), n_cross=int(len(idx)),
                min_abs_dpsi_at_cross=float(np.min(np.abs(dpsi[idx]))) if len(idx) else None,
                nodes=int(sol.x.size), rms_res=float(np.max(sol.rms_residuals)),
                y0=[float(v) for v in np.exp(sol.sol(0.0))])


def continue_branch(tol=1e-8, direction=-1, ds0=0.02, n_steps=400, b_stop=(5e-5, 2.6e-3),
                    log=print, stop_after_folds=2, stop_near_bcrit=True):
    br = Branch(tol=tol)
    # seed: attracting cycle at b = 0.5 bcrit from direct simulation
    b0 = 0.5 * BCRIT
    eq, period, frac, tt, yy, psi = simulate_cycle(b0, BCRIT)
    sgrid = (tt - tt[0]) / (tt[-1] - tt[0])
    sel = np.linspace(0, len(sgrid) - 1, 1500).astype(int)
    sgrid, U = sgrid[sel], np.log(yy[:, sel])
    U[:, -1] = U[:, 0]
    sol, T, b = br.solve(sgrid, U, period, b0, fixed_b=True)
    assert sol.success, sol.message
    pts = [(sol, T, b)]
    # first arclength step along the parameter direction
    z1 = np.r_[sol.sol(0.0), T / TS, b / BS]
    t0vec = np.zeros_like(z1); t0vec[-1] = direction
    sol2, T2, b2 = br.solve(sol.x, sol.y, T, b * (1 + direction * 0.01),
                            z_prev=z1, tangent=t0vec, ds=abs(b * 0.01 / BS))
    assert sol2.success, sol2.message
    pts.append((sol2, T2, b2))
    rows = [orbit_summary(*pts[0]), orbit_summary(*pts[1])]
    ds = ds0
    folds = 0
    k = 0
    while k < n_steps:
        (s1, T1, b1), (s2, T2, b2) = pts[-2], pts[-1]
        z1 = np.r_[s1.sol(0.0), T1 / TS, b1 / BS]
        z2 = np.r_[s2.sol(0.0), T2 / TS, b2 / BS]
        tvec = z2 - z1
        tvec /= np.linalg.norm(tvec)
        # predictor: secant extrapolation of the whole profile (common mesh)
        xg = np.unique(np.r_[np.linspace(0, 1, 2500), s2.x[::max(1, s2.x.size // 2500)]])
        Upred = s2.sol(xg) + (ds / np.linalg.norm(z2 - z1)) * (s2.sol(xg) - s1.sol(xg))
        Tpred = T2 + (ds / np.linalg.norm(z2 - z1)) * (T2 - T1)
        bpred = b2 + (ds / np.linalg.norm(z2 - z1)) * (b2 - b1)
        sol, T, b = br.solve(xg, Upred, Tpred, bpred, z_prev=z2, tangent=tvec, ds=ds)
        if not sol.success or not (20 < T < 2000):
            ds *= 0.5
            if ds < 1e-6:
                log("step size underflow; stopping")
                break
            continue
        pts.append((sol, T, b))
        row = orbit_summary(sol, T, b)
        rows.append(row)
        k += 1
        # turning point detection in b
        if len(rows) >= 3:
            bA, bB, bC = rows[-3]["b"], rows[-2]["b"], rows[-1]["b"]
            if (bB - bA) * (bC - bB) < 0:
                folds += 1
                log(f"  turning point bracketed near b={bB:.8e}")
        log(f"{k:4d} b={b:.8e} T={T:.4f} ampq={row['ampq']:.5e} psi=[{row['psimin']:.2e},{row['psimax']:.2e}] "
            f"ncross={row['n_cross']} nodes={row['nodes']} ds={ds:.3g}")
        if folds >= stop_after_folds and stop_near_bcrit and b < 1.002 * BCRIT and rows[-1]['ampq'] < 2e-3:
            break
        if b < b_stop[0] or b > b_stop[1] or rows[-1]["ampq"] < 2e-5:
            break
        if sol.x.size < 6000:
            ds = min(ds * 1.3, 0.2)
        pts = pts[-3:]
    return rows


def locate_folds(rows):
    """Quadratic interpolation of b against arclength index at each turning point."""
    out = []
    bs = np.array([r["b"] for r in rows])
    # arclength in (log amplitude, b) for interpolation
    for j in range(1, len(rows) - 1):
        if (bs[j] - bs[j - 1]) * (bs[j + 1] - bs[j]) < 0:
            x = np.array([rows[j - 1]["ampq"], rows[j]["ampq"], rows[j + 1]["ampq"]])
            y = bs[j - 1:j + 2]
            cf = np.polyfit(x, y, 2)
            xv = -cf[1] / (2 * cf[0])
            out.append(dict(index=j, b_fold=float(np.polyval(cf, xv)), ampq_fold=float(xv),
                            b_grid_extreme=float(bs[j]), row=rows[j]))
    return out


if __name__ == "__main__":
    tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
    direction = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    t0 = time.time()
    rows = continue_branch(tol=tol, direction=direction)
    folds = locate_folds(rows)
    json.dump(dict(tol=tol, direction=direction, rows=rows, folds=folds, cpu_s=time.time() - t0),
              open(OUT / f"collocation_branch_tol{tol:g}_dir{direction}.json", "w"), indent=1)
    for f in folds:
        print("FOLD", f["b_fold"], f["ampq_fold"])
