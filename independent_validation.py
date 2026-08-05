#!/usr/bin/env python3
"""Independent reproduction of the Hopf coefficient and global cycle.

Independent validator: Suleiman Ibrahim.

This script is deliberately independent of verify_revision.py:
  * derivatives are generated symbolically with SymPy and evaluated with
    80-digit mpmath arithmetic;
  * the large attracting cycle is integrated with a fixed-step classical RK4
    method, rather than an adaptive SciPy solver.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.linalg import eigvals
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent
P = {
    "B": 3.0, "m": 0.002, "eta": 0.5, "pi": 0.20,
    "a": 0.70, "c": 2.0, "chiQ": 0.15, "chiH": 0.25,
    "rhoI": 0.788, "rhoQ": 0.05, "rhoH": 0.20, "theta": 0.20,
}


def R0_value() -> float:
    p = P
    return p["B"] * p["eta"] / (p["eta"] + p["m"]) * (
        1.0 + p["theta"] * p["pi"] / (p["a"] + p["chiQ"])
    )


def normal_equilibrium() -> tuple[np.ndarray, float]:
    p = P
    R0 = R0_value()
    KQ = p["pi"] / (p["a"] + p["chiQ"])
    KH = p["a"] * p["pi"] / ((p["a"] + p["chiQ"]) * p["chiH"])
    KR = (p["rhoI"] + p["rhoQ"] * KQ + p["rhoH"] * KH) / p["m"]
    G = 1.0 / p["eta"] + 1.0 + KQ + KH + KR
    L = (p["eta"] + p["m"]) / (p["m"] * p["eta"])
    ii = (R0 - 1.0) / (G + L * (R0 - 1.0))
    y = np.array([1.0 - L * ii, ii / p["eta"], ii, KQ * ii, KH * ii, KR * ii])
    bcrit = y[4] + p["a"] / p["c"] * y[3]
    return y, float(bcrit)


def capacity_equilibrium(b: float) -> np.ndarray:
    p = P
    L = (p["eta"] + p["m"]) / (p["m"] * p["eta"])
    hh = p["c"] / (p["c"] + p["chiH"]) * b
    AA = p["c"] * p["chiH"] / (p["c"] + p["chiH"]) * b

    def qq(ii: float) -> float:
        return (p["pi"] * ii - AA) / p["chiQ"]

    def ss(ii: float) -> float:
        return 1.0 - L * ii

    def rr(ii: float) -> float:
        return (p["rhoI"] * ii + p["rhoQ"] * qq(ii) + p["rhoH"] * hh) / p["m"]

    def residual(ii: float) -> float:
        nn = ss(ii) + ii / p["eta"] + ii + qq(ii) + hh + rr(ii)
        return p["B"] * ss(ii) * (ii + p["theta"] * qq(ii)) - (
            (p["eta"] + p["m"]) / p["eta"] * ii * nn
        )

    lo = AA * (p["a"] + p["chiQ"]) / (p["a"] * p["pi"])
    hi = (1.0 / L) * (1.0 - 1e-12)
    ii = brentq(residual, lo, hi, xtol=1e-14)
    return np.array([ss(ii), ii / p["eta"], ii, qq(ii), hh, rr(ii)])


def jacobian_capacity(y: np.ndarray, b: float) -> np.ndarray:
    p = P
    s, e, i, q, h, r = y
    n = float(np.sum(y))
    u = i + p["theta"] * q
    Ts = p["B"] * u * (n - s) / n**2
    Te = -p["B"] * s * u / n**2
    Ti = p["B"] * s * (n - u) / n**2
    Tq = p["B"] * s * (p["theta"] * n - u) / n**2
    Th = -p["B"] * s * u / n**2
    Tr = Th
    return np.array([
        [-p["m"] - Ts, -Te, -Ti, -Tq, -Th, -Tr],
        [Ts, Te - (p["eta"] + p["m"]), Ti, Tq, Th, Tr],
        [0, p["eta"], -1, 0, 0, 0],
        [0, 0, p["pi"], -p["chiQ"], p["c"], 0],
        [0, 0, 0, 0, -(p["c"] + p["chiH"]), 0],
        [0, 0, p["rhoI"], p["rhoQ"], p["rhoH"], -p["m"]],
    ], dtype=float)


def spectral_abscissa_capacity(b: float) -> float:
    y = capacity_equilibrium(b)
    return float(np.max(eigvals(jacobian_capacity(y, b)).real))


def find_hopf(bcrit: float) -> tuple[float, np.ndarray, float, float]:
    grid = np.geomspace(1e-7, bcrit * (1.0 - 1e-9), 600)
    vals = np.array([spectral_abscissa_capacity(x) for x in grid])
    idx = np.where(vals[:-1] * vals[1:] < 0.0)[0]
    if len(idx) != 1:
        raise RuntimeError(f"Expected one crossing, found {len(idx)}")
    j = int(idx[0])
    bH = brentq(spectral_abscissa_capacity, grid[j], grid[j + 1], xtol=1e-15)
    yH = capacity_equilibrium(bH)
    ev = eigvals(jacobian_capacity(yH, bH))
    omega = float(np.max(ev.imag))
    db = 1e-8
    trans = (spectral_abscissa_capacity(bH + db) - spectral_abscissa_capacity(bH - db)) / (2 * db)
    return float(bH), yH, omega, float(trans)


def first_lyapunov_sympy_mpmath(yH: np.ndarray, bH: float) -> float:
    mp.mp.dps = 80
    s, e, i, q, h, r = sp.symbols("s e i q h r", real=True)
    ys = [s, e, i, q, h, r]
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta, b = sp.symbols(
        "B m eta pi a c chiQ chiH rhoI rhoQ rhoH theta b", real=True
    )
    n = sum(ys)
    incidence = B * s * (i + theta * q) / n
    admission = c * (b - h)
    f = sp.Matrix([
        m * (1 - s) - incidence,
        incidence - (eta + m) * e,
        eta * e - i,
        pi * i - admission - chiQ * q,
        admission - chiH * h,
        rhoI * i + rhoQ * q + rhoH * h - m * r,
    ])
    J = f.jacobian(ys)
    H = [sp.hessian(f[k], ys) for k in range(6)]
    C = [[[[sp.diff(f[kk], ys[j], ys[l], ys[z]) for z in range(6)]
            for l in range(6)] for j in range(6)] for kk in range(6)]
    subs = {
        B:P["B"], m:P["m"], eta:P["eta"], pi:P["pi"], a:P["a"], c:P["c"],
        chiQ:P["chiQ"], chiH:P["chiH"], rhoI:P["rhoI"], rhoQ:P["rhoQ"],
        rhoH:P["rhoH"], theta:P["theta"], b:bH,
    }
    subs.update({ys[k]:float(yH[k]) for k in range(6)})

    def mpnum(expr):
        return mp.mpf(str(sp.N(expr.subs(subs), 70)))

    A = mp.matrix([[mpnum(J[j,k]) for k in range(6)] for j in range(6)])
    Hn = [[[mpnum(H[comp][j,k]) for k in range(6)] for j in range(6)] for comp in range(6)]
    Cn = [[[[mpnum(C[comp][j][k][l]) for l in range(6)] for k in range(6)]
           for j in range(6)] for comp in range(6)]
    E, Lrows, Rcols = mp.eig(A, left=True, right=True)
    kp = max(range(6), key=lambda k: mp.im(E[k]))
    km = min(range(6), key=lambda k: mp.im(E[k]))
    omega = mp.im(E[kp])
    qv = mp.matrix([Rcols[j,kp] for j in range(6)])
    qv /= mp.sqrt(sum(mp.conj(qv[j]) * qv[j] for j in range(6)))
    pv = mp.matrix([Lrows[km,j] for j in range(6)])
    inner = sum(mp.conj(pv[j]) * qv[j] for j in range(6))
    pv /= mp.conj(inner)

    def Bform(x, y):
        return mp.matrix([
            sum(Hn[comp][j][k] * x[j] * y[k] for j in range(6) for k in range(6))
            for comp in range(6)
        ])

    def Cform(x, y, z):
        return mp.matrix([
            sum(Cn[comp][j][k][l] * x[j] * y[k] * z[l]
                for j in range(6) for k in range(6) for l in range(6))
            for comp in range(6)
        ])

    qbar = mp.matrix([mp.conj(x) for x in qv])
    term = (
        Cform(qv, qv, qbar)
        - 2 * Bform(qv, mp.lu_solve(A, Bform(qv, qbar)))
        + Bform(qbar, mp.lu_solve(2j * omega * mp.eye(6) - A, Bform(qv, qv)))
    )
    g21 = sum(mp.conj(pv[j]) * term[j] for j in range(6))
    return float(mp.re(g21) / (2 * omega))


def rhs_full(y: np.ndarray, b: float) -> np.ndarray:
    p = P
    s, e, i, q, h, r = y
    n = float(np.sum(y))
    incidence = p["B"] * s * (i + p["theta"] * q) / n
    admission = min(p["a"] * q, p["c"] * max(b - h, 0.0))
    return np.array([
        p["m"] * (1 - s) - incidence,
        incidence - (p["eta"] + p["m"]) * e,
        p["eta"] * e - i,
        p["pi"] * i - admission - p["chiQ"] * q,
        admission - p["chiH"] * h,
        p["rhoI"] * i + p["rhoQ"] * q + p["rhoH"] * h - p["m"] * r,
    ])


def cycle_fixed_rk4(b: float, bcrit: float, dt: float = 0.05, t_end: float = 12000.0):
    eq = capacity_equilibrium(b)
    y = eq.copy()
    y[2] *= 1.0 + 1e-6
    store_start = 7000.0
    ts, qs, hs, psis = [], [], [], []
    steps = int(round(t_end / dt))
    for k in range(steps):
        k1 = rhs_full(y, b)
        k2 = rhs_full(y + 0.5 * dt * k1, b)
        k3 = rhs_full(y + 0.5 * dt * k2, b)
        k4 = rhs_full(y + dt * k3, b)
        y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t = (k + 1) * dt
        if t >= store_start:
            ts.append(t); qs.append(y[3]); hs.append(y[4])
            psis.append(P["a"] * y[3] - P["c"] * (b - y[4]))
    ts = np.asarray(ts); qs = np.asarray(qs); hs = np.asarray(hs); psis = np.asarray(psis)
    peaks = np.where((qs[1:-1] > qs[:-2]) & (qs[1:-1] >= qs[2:]))[0] + 1
    if len(peaks) < 10:
        raise RuntimeError("Cycle not resolved")
    tp = ts[peaks]
    period = float(np.mean(np.diff(tp[-9:])))
    t1, t2 = tp[-2], tp[-1]
    mask = (ts >= t1) & (ts < t2)
    fraction = float(np.mean(psis[mask] >= 0.0))
    crossings = int(np.sum(psis[mask][:-1] * psis[mask][1:] < 0.0))
    return {
        "integration_method": "fixed-step classical RK4",
        "dt": dt,
        "t_end": t_end,
        "b_cycle": b,
        "b_cycle_over_bcrit": b / bcrit,
        "period": period,
        "capacity_limited_fraction": fraction,
        "switching_crossings_per_period": crossings,
        "q_min": float(np.min(qs[mask])),
        "q_max": float(np.max(qs[mask])),
        "h_min": float(np.min(hs[mask])),
        "h_max": float(np.max(hs[mask])),
    }


def main():
    _, bcrit = normal_equilibrium()
    bH, yH, omega, trans = find_hopf(bcrit)
    l1 = first_lyapunov_sympy_mpmath(yH, bH)
    cycle = cycle_fixed_rk4(0.5 * bcrit, bcrit)
    result = {
        "implementation": "independent SymPy/mpmath derivatives plus fixed-step RK4",
        "R0": R0_value(),
        "bcrit": bcrit,
        "bH": bH,
        "bH_over_bcrit": bH / bcrit,
        "omegaH": omega,
        "transversality_dReLambda_db": trans,
        "first_lyapunov_coefficient": l1,
        "hopf_type_under_standard_Kuznetsov_convention": "subcritical" if l1 > 0 else "supercritical",
        "cycle": cycle,
    }
    (OUT / "independent_validation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
