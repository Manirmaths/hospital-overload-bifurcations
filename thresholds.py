#!/usr/bin/env python3
"""Certified computation of the transient capacity threshold b_trans.

Revision 1.1.0 replacement for ``transient_threshold`` in verify_revision.py.

Definition (manuscript Eq. 25):  b_trans = sup_{tau >= 0} P(tau),
P(tau) = h_N(tau) + (a/c) q_N(tau) along the normal-care trajectory.

Version 1.0.0 searched only tau in [0, 100].  This module searches the whole
half-line by

1. integrating the normal-care system on successively doubled horizons;
2. locating every interior local maximum of P on a fine grid and refining each
   with a bounded scalar optimiser (the global maximum is the largest of them,
   or P(0) if larger);
3. stopping only when a *tail certificate* holds: the trajectory is inside a
   small neighbourhood of the normal-care equilibrium E_N, and the
   linearised modal bound
       |P(tau) - b_crit| <= sum_k |w^T v_k| |u_k^T (y(T) - E_N)|,  tau >= T,
   (v_k, u_k right/left eigenvectors of J_N, Re lambda_k < 0) plus a
   quadratic safety allowance is smaller than (max_found - b_crit) / 2.
   Then no later time can exceed the maximum already found.

The returned diagnostics record the horizon, number of local maxima examined,
the certificate margin and the theorem check b_trans >= b_crit (Theorem 7.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.linalg import eig

from verify_revision import P0, Y0, rhs_normal, normal_equilibrium, jacobian, R0_value


@dataclass
class TransientResult:
    btrans: float
    tau_star: float
    bcrit: float
    horizon: float
    n_local_maxima: int
    certified: bool
    tail_bound: float
    local_maxima: list = field(default_factory=list)
    sol: object = None

    @property
    def ratio(self):
        return self.btrans / self.bcrit


def _tail_bound(y_end, yN, JN, w):
    """Modal bound on sup_{tau>=T} |P(tau)-b_crit| for the linearisation."""
    lam, VL, VR = eig(JN, left=True, right=True)
    # normalise so that u_k^T v_k = 1
    U = VL.conj()
    scale = np.array([U[:, k] @ VR[:, k] for k in range(len(lam))])
    d = y_end - yN
    total = 0.0
    for k in range(len(lam)):
        if lam[k].real >= 0:
            return np.inf
        total += abs(w @ VR[:, k]) * abs(U[:, k] @ d / scale[k])
    return float(total)


def transient_threshold_certified(p=P0, y0=Y0, rtol=1e-10, atol=1e-14,
                                  T0=200.0, T_max=2.0e5, dt=0.05,
                                  nbhd=1e-2, quad_factor=10.0):
    """Return a TransientResult for b_trans = sup_{tau>=0} P(tau).

    Raises RuntimeError if R0 <= 1 (no endemic normal-care equilibrium) or
    if the tail certificate cannot be obtained before T_max.
    """
    p = np.asarray(p, dtype=float)
    a, c = p[4], p[5]
    if R0_value(p) <= 1.0:
        raise RuntimeError("R0 <= 1: b_crit undefined")
    yN, bcrit = normal_equilibrium(p)
    JN = jacobian(yN, bcrit, "N", p)
    w = np.zeros(6)
    w[3], w[4] = a / c, 1.0

    def P_of(sol, t):
        y = sol.sol(t)
        return y[4] + a / c * y[3]

    T = T0
    while True:
        sol = solve_ivp(lambda t, y: rhs_normal(t, y, p), (0.0, T), y0,
                        method="DOP853", rtol=rtol, atol=atol,
                        dense_output=True, max_step=1.0)
        if not sol.success:
            raise RuntimeError(sol.message)
        dtT = max(dt, T / 1.0e5)
        ts = np.arange(0.0, T + 0.5 * dtT, dtT)
        Pv = P_of(sol, ts)
        idx = np.where((Pv[1:-1] >= Pv[:-2]) & (Pv[1:-1] > Pv[2:]))[0] + 1
        cands = []
        for j in idx:
            opt = minimize_scalar(lambda t: -P_of(sol, t), bounds=(ts[j - 1], ts[j + 1]),
                                  method="bounded", options={"xatol": 1e-11})
            cands.append((float(-opt.fun), float(opt.x)))
        cands.append((float(Pv[0]), 0.0))
        best = max(cands)
        # tail certificate at the end of the current horizon
        y_end = sol.y[:, -1]
        rel = np.linalg.norm(y_end - yN) / np.linalg.norm(yN)
        tb = _tail_bound(y_end, yN, JN, w)
        tb_safe = tb * (1.0 + quad_factor * rel)
        gap = best[0] - bcrit
        certified = rel < nbhd and gap > 0 and tb_safe < 0.5 * gap
        if certified or T >= T_max:
            break
        T *= 2.0
    if not certified:
        raise RuntimeError(f"tail certificate failed up to T={T}")
    cands.sort(reverse=True)
    return TransientResult(btrans=best[0], tau_star=best[1], bcrit=float(bcrit),
                           horizon=float(T), n_local_maxima=len(idx), certified=True,
                           tail_bound=tb_safe, local_maxima=cands[:6], sol=sol)


def check_threshold_ordering(res: TransientResult, tol=1e-12):
    """Automated consistency check of Theorem 7.1: b_trans >= b_crit."""
    if not res.btrans >= res.bcrit * (1.0 - tol):
        raise AssertionError(
            f"Theorem 7.1 violated: b_trans={res.btrans:.6e} < b_crit={res.bcrit:.6e}")
    return True


if __name__ == "__main__":
    for B in [3.0, 1.13889]:
        p = P0.copy(); p[0] = B
        r = transient_threshold_certified(p)
        check_threshold_ordering(r)
        print(f"B={B}: btrans={r.btrans:.10e} tau*={r.tau_star:.5f} bcrit={r.bcrit:.6e} "
              f"ratio={r.ratio:.4f} horizon={r.horizon:.0f} nmax={r.n_local_maxima} "
              f"tail={r.tail_bound:.2e}")
