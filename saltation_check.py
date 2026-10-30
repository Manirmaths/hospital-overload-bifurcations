#!/usr/bin/env python3
"""R1.1 / R2.2: treatment of switching-surface crossings in the variational
equations, checked on cycles adjacent to both folds.

For a continuous piecewise-smooth vector field (f_C - f_N = Psi w), the
saltation matrix at a transversal crossing,
    S = I + (f_+ - f_-) n^T / (n^T f_-),   n = grad Psi,
reduces to the identity because f_+ = f_- on Sigma.  The monodromy matrix is
therefore obtained by integrating the variational equations with the
one-sided Jacobian of the active region, which is what v1.0.0 does.
We verify this three ways on the same cycle:
  (a) v1.0.0 method: DOP853, regime chosen pointwise, no event location;
  (b) event-located segments, explicit saltation matrices at each crossing;
  (c) central finite differences of the time-T flow map (Radau, rtol 1e-12).
Crossing transversality |dPsi/dtau| and the distance of the cycle from
grazing (min over the N-segment of -Psi, max over the C-segment of Psi) are
also reported.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import eigvals

from verify_revision import P0, normal_equilibrium, capacity_equilibria, jacobian, rhs_full

OUT = Path(__file__).resolve().parent / "results"
B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, P0)
_, BCRIT = normal_equilibrium()
nvec = np.array([0, 0, 0, a, c, 0.0])


def psi(y, b):
    return a * y[3] - c * (b - y[4])


def f_reg(y, b, reg):
    s, e, i, q, h, r = y
    n = y.sum(); inc = B * s * (i + theta * q) / n
    A = a * q if reg == "N" else c * (b - h)
    return np.array([m * (1 - s) - inc, inc - (eta + m) * e, eta * e - i,
                     pi * i - A - chiQ * q, A - chiH * h, rhoI * i + rhoQ * q + rhoH * h - m * r])


def converge_cycle(b, y0, t_settle=60000.0):
    sol = solve_ivp(lambda t, y: rhs_full(t, y, b), (0, t_settle), y0, method="LSODA",
                    rtol=1e-11, atol=1e-14, dense_output=True)
    tt = np.linspace(t_settle - 800, t_settle, 80001)
    yy = sol.sol(tt)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(yy[3])
    T = float(np.mean(np.diff(tt[pk])))
    x = yy[:, pk[-1]]
    # Newton polish on (x, T) with phase condition q'(0)=0 using FD flow map
    def F(z):
        x, T = z[:6], z[6]
        ye = solve_ivp(lambda t, y: rhs_full(t, y, b), (0, T), x, method="DOP853",
                       rtol=1e-12, atol=1e-16).y[:, -1]
        return np.r_[(ye - x) / np.r_[0.1, 0.01, 0.01, 0.01, 0.001, 1.0], rhs_full(0, x, b)[3] / 1e-5]
    r = root(F, np.r_[x, T], method="hybr", options={"xtol": 1e-12})
    return r.x[:6], float(r.x[6]), float(np.max(np.abs(F(r.x))))


def monodromy_a(x, T, b):
    def aug(t, z):
        y = z[:6]; M = z[6:].reshape(6, 6)
        reg = "N" if psi(y, b) < 0 else "C"
        return np.r_[rhs_full(t, y, b), (jacobian(y, b, reg) @ M).ravel()]
    z = solve_ivp(aug, (0, T), np.r_[x, np.eye(6).ravel()], method="DOP853",
                  rtol=1e-11, atol=1e-14, max_step=0.3).y[:, -1]
    return z[6:].reshape(6, 6)


def monodromy_b(x, T, b):
    """Event-located segments with explicit saltation matrices."""
    t0, y = 0.0, x.copy(); M = np.eye(6)
    reg = "N" if psi(y, b) < 0 else "C"
    crossings = []
    while t0 < T - 1e-12:
        def ev(t, z): return psi(z[:6], b)
        ev.terminal = True; ev.direction = 1 if reg == "N" else -1
        def aug(t, z, reg=reg):
            yy = z[:6]; MM = z[6:].reshape(6, 6)
            return np.r_[f_reg(yy, b, reg), (jacobian(yy, b, reg) @ MM).ravel()]
        sol = solve_ivp(aug, (t0, T), np.r_[y, M.ravel()], method="DOP853", rtol=1e-12, atol=1e-15,
                        events=ev)
        z = sol.y[:, -1]
        y, M = z[:6], z[6:].reshape(6, 6)
        if sol.status == 1 and sol.t_events[0][0] > t0 + 1e-9:
            t0 = float(sol.t_events[0][0])
            new = "C" if reg == "N" else "N"
            fm, fp = f_reg(y, b, reg), f_reg(y, b, new)
            S = np.eye(6) + np.outer(fp - fm, nvec) / (nvec @ fm)
            M = S @ M
            crossings.append(dict(t=t0, direction=f"{reg}->{new}", dpsi_dtau=float(nvec @ fm),
                                  saltation_minus_I=float(np.max(np.abs(S - np.eye(6)))),
                                  f_jump=float(np.max(np.abs(fp - fm)))))
            reg = new
        else:
            t0 = float(sol.t[-1])
    return M, crossings


def monodromy_c(x, T, b, h=1e-7):
    scale = np.array([0.1, 0.01, 0.01, 0.01, 0.001, 1.0])
    M = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6); d[j] = h * scale[j]
        yp = solve_ivp(lambda t, y: rhs_full(t, y, b), (0, T), x + d, method="Radau", rtol=1e-12, atol=1e-16).y[:, -1]
        ym = solve_ivp(lambda t, y: rhs_full(t, y, b), (0, T), x - d, method="Radau", rtol=1e-12, atol=1e-16).y[:, -1]
        M[:, j] = (yp - ym) / (2 * d[j])
    return M


def nontrivial(M):
    v = eigvals(M)
    k = int(np.argmin(np.abs(v - 1)))
    rest = np.delete(v, k)
    return sorted(rest, key=lambda z: -abs(z))


def main():
    out = []
    # seeds: attracting cycle continued from 0.5 bcrit
    from bistability_demo import cycle_point
    for b in [8.70e-5, 9.0e-5, 1.5e-4, 1.5e-3, 2.05e-3, 2.08e-3]:
        y0 = cycle_point(b)
        x, T, res = converge_cycle(b, y0)
        Ma = monodromy_a(x, T, b); Mb, cr = monodromy_b(x, T, b); Mc = monodromy_c(x, T, b)
        mu = {k: nontrivial(M)[:2] for k, M in zip("abc", (Ma, Mb, Mc))}
        tt = np.linspace(0, T, 20001)
        yy = solve_ivp(lambda t, y: rhs_full(t, y, b), (0, T), x, method="DOP853", rtol=1e-12,
                       atol=1e-15, dense_output=True).sol(tt)
        ps = a * yy[3] - c * (b - yy[4])
        rec = dict(b=b, b_over_bcrit=b / BCRIT, T=T, shooting_residual=res,
                   psi_min=float(ps.min()), psi_max=float(ps.max()), frac_C=float(np.mean(ps >= 0)),
                   crossings=cr,
                   mu_a=[[z.real, z.imag] for z in mu["a"]], mu_b=[[z.real, z.imag] for z in mu["b"]],
                   mu_c=[[z.real, z.imag] for z in mu["c"]],
                   max_abs_diff_ab=float(np.max(np.abs(Ma - Mb))), max_abs_diff_ac=float(np.max(np.abs(Ma - Mc)) / np.max(np.abs(Ma))))
        out.append(rec)
        print(f"b={b:.3e} T={T:.4f} res={res:.1e} |mu1| a={abs(mu['a'][0]):.6f} b={abs(mu['b'][0]):.6f} "
              f"c={abs(mu['c'][0]):.6f} ncross={len(cr)} |dPsi|min={min(abs(x['dpsi_dtau']) for x in cr):.3e} "
              f"S-I={max(x['saltation_minus_I'] for x in cr):.1e} psi=[{ps.min():.2e},{ps.max():.2e}]", flush=True)
    json.dump(out, open(OUT / "saltation_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
