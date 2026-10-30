#!/usr/bin/env python3
"""R2.8: structural robustness of (i) the subcritical Hopf point, (ii) the two
folds of periodic orbits, and (iii) the separation b_trans >> b_crit.

Folds are bracketed by *hysteresis simulation*, independent of both shooting
and collocation: starting on the attracting cycle at a seed capacity inside
(b_H, b_crit), b is moved in small steps, each run started from the cycle
reached at the previous step; the first b at which the trajectory relaxes to
the equilibrium brackets the fold, and the bracket is then bisected.

Usage: python robustness.py oat      one-at-a-time +/-25% variations
       python robustness.py Bcurve   fold brackets along the B axis (atlas)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.linalg import eigvals
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.optimize import brentq

from verify_revision import P0, NAMES, R0_value, normal_equilibrium, capacity_equilibria, jacobian
from thresholds import transient_threshold_certified, check_threshold_ordering

OUT = Path(__file__).resolve().parent / "results"
T_RUN = 15000.0
AMP_TOL = 1e-4


def rhs(t, y, b, p):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = p
    s, e, i, q, h, r = y
    n = s + e + i + q + h + r
    inc = B * s * (i + theta * q) / n
    A = min(a * q, c * max(b - h, 0.0))
    return [m * (1 - s) - inc, inc - (eta + m) * e, eta * e - i,
            pi * i - A - chiQ * q, A - chiH * h, rhoI * i + rhoQ * q + rhoH * h - m * r]


def settle(y0, b, p, t_end=T_RUN):
    pt = tuple(float(x) for x in p)
    sol = solve_ivp(rhs, (0, t_end), y0, args=(b, pt), method="LSODA", rtol=1e-9, atol=1e-13,
                    dense_output=True)
    tt = np.linspace(t_end - 1500, t_end, 6001)
    yy = sol.sol(tt)
    amp = float(np.ptp(yy[3]))
    j = int(np.argmax(yy[3]))
    return amp, yy[:, j]


def hopf_points(p):
    _, bc = normal_equilibrium(p)

    def alphaC(b):
        roots = capacity_equilibria(b, p)
        if not roots:
            return np.nan
        return float(np.max(eigvals(jacobian(roots[0], b, "C", p)).real))
    grid = np.geomspace(bc * 1e-4, bc * (1 - 1e-8), 150)
    vals = np.array([alphaC(x) for x in grid])
    idx = np.where(vals[:-1] * vals[1:] < 0)[0]
    return [brentq(alphaC, grid[k], grid[k + 1], xtol=1e-16) for k in idx]


def lyapunov_l1(p, bH):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, p)
    yH = capacity_equilibria(bH, p)[0]

    def f(y):
        s, e, i, q, h, r = y
        n = jnp.sum(y)
        inc = B * s * (i + theta * q) / n
        A = c * (bH - h)
        return jnp.array([m * (1 - s) - inc, inc - (eta + m) * e, eta * e - i,
                          pi * i - A - chiQ * q, A - chiH * h, rhoI * i + rhoQ * q + rhoH * h - m * r])
    yj = jnp.array(yH)
    A = np.array(jax.jacfwd(f)(yj))
    H = np.array(jax.jacfwd(jax.jacfwd(f))(yj))
    C3 = np.array(jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(yj))
    vals, vl, vr = eig(A, left=True, right=True)
    k = int(np.argmax(vals.imag))
    q = vr[:, k]; pv = vl[:, k]; pv = pv / np.conj(np.vdot(pv, q)); om = vals[k].imag
    Bf = lambda x, y: np.einsum("ijk,j,k->i", H, x, y)
    Cf = lambda x, y, z: np.einsum("ijkl,j,k,l->i", C3, x, y, z)
    g21 = np.vdot(pv, Cf(q, q, np.conj(q)) - 2 * Bf(q, np.linalg.solve(A, Bf(q, np.conj(q))))
                  + Bf(np.conj(q), np.linalg.solve(2j * om * np.eye(6) - A, Bf(q, q))))
    return float(g21.real / (2 * om)), float(om)


def fold_bracket(p, b_seed, y_seed, direction, b_limit, rel_tol=2e-3, log=None):
    """March b from b_seed in geometric steps (direction -1 down, +1 up) staying on
    the attracting cycle; return bracket (b_on_cycle, b_off_cycle) for the fold."""
    b, y = b_seed, y_seed
    factor = 1.12 if direction > 0 else 1 / 1.12
    last_on, y_on = b, y
    while True:
        b_new = b * factor
        if (direction < 0 and b_new < b_limit) or (direction > 0 and b_new > b_limit):
            return dict(on=last_on, off=None, note="cycle persists to search limit")
        amp, y_new = settle(y_on, b_new, p)
        if amp > AMP_TOL:
            last_on, y_on, b = b_new, y_new, b_new
        else:
            off = b_new
            break
    lo, hi = last_on, off
    while abs(hi - lo) / lo > rel_tol:
        mid = np.sqrt(lo * hi)
        amp, y_mid = settle(y_on, mid, p)
        if amp > AMP_TOL:
            lo, y_on = mid, y_mid
        else:
            hi = mid
    return dict(on=float(lo), off=float(hi), estimate=float(np.sqrt(lo * hi)))


def analyse(p, label=""):
    p = np.asarray(p, float)
    res = dict(label=label, params=dict(zip(NAMES, map(float, p))), R0=float(R0_value(p)))
    if res["R0"] <= 1:
        res["note"] = "R0<=1"; return res
    _, bc = normal_equilibrium(p)
    res["bcrit"] = float(bc)
    tr = transient_threshold_certified(p)
    check_threshold_ordering(tr)
    res["btrans"] = tr.btrans; res["btrans_over_bcrit"] = tr.ratio
    hb = hopf_points(p)
    res["bH"] = [float(x) for x in hb]
    if len(hb) != 1:
        res["note"] = f"{len(hb)} Hopf points on capacity branch"; return res
    bH = hb[0]
    l1, om = lyapunov_l1(p, bH)
    res.update(l1=l1, omegaH=om, hopf="subcritical" if l1 > 0 else "supercritical",
               bH_over_bcrit=bH / bc)
    b_seed = np.sqrt(bH * bc)
    eq = capacity_equilibria(b_seed, p)[0]
    y0 = eq.copy(); y0[2] *= 1.001
    amp, yc = settle(y0, b_seed, p, t_end=30000.0)
    res["seed_amp"] = amp
    if amp < AMP_TOL:
        res["note"] = "no attracting cycle at seed"; return res
    lo = fold_bracket(p, b_seed, yc, -1, b_limit=bH * 1e-3)
    up = fold_bracket(p, b_seed, yc, +1, b_limit=bc * 50)
    res["fold_lower"] = lo; res["fold_upper"] = up
    if lo.get("off") and up.get("off"):
        res["bistable_lower"] = [lo["estimate"], bH]
        res["bistable_upper"] = [bc, up["estimate"]]
        res["lower_fold_below_bH"] = bool(lo["estimate"] < bH)
        res["upper_fold_above_bcrit"] = bool(up["estimate"] > bc)
    return res


def _job(args):
    p, label = args
    try:
        r = analyse(p, label)
    except Exception as exc:  # record failures rather than abort the sweep
        r = dict(label=label, error=repr(exc))
    print(label, {k: r.get(k) for k in ("R0", "btrans_over_bcrit", "bH_over_bcrit", "l1", "hopf")},
          r.get("fold_lower", {}).get("estimate"), r.get("fold_upper", {}).get("estimate"), flush=True)
    return r


def oat_sets(frac=0.25):
    sets = [(P0.copy(), "baseline")]
    for name in ["B", "eta", "pi", "a", "c", "chiQ", "chiH", "thetaQ", "rhoH"]:
        j = NAMES.index(name)
        for sgn in (-1, 1):
            p = P0.copy(); p[j] *= 1 + sgn * frac
            # keep the scaling identities: pi + rhoI + m + Delta_I = 1 (adjust rhoI);
            # chi_Q, chi_H are total removal rates, so Delta_Q, Delta_H absorb changes.
            if name == "pi":
                p[8] = 1 - p[3] - p[1] - 0.010
            if name == "rhoH":
                p[7] = p[10] + 0.048 + p[1]
            sets.append((p, f"{name}{'+' if sgn > 0 else '-'}{int(frac * 100)}%"))
    return sets


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "oat"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if mode == "oat":
        jobs = oat_sets()
        fname = "robustness_oat.json"
    else:
        jobs = []
        for B in [1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
            p = P0.copy(); p[0] = B
            jobs.append((p, f"B={B}"))
        fname = "robustness_Bcurve.json"
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_job, jobs))
    json.dump(out, open(OUT / fname, "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
