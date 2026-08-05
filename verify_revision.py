#!/usr/bin/env python3
"""Reproduce the principal numerical checks and figures for the revised manuscript.

Dependencies: numpy, scipy, matplotlib. JAX is used only for the first Lyapunov
coefficient; the remaining computations do not require it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.linalg import eigvals, det, norm
from scipy.integrate import solve_ivp, quad
from scipy.linalg import expm, eig
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# Baseline dimensionless parameters
NAMES = ["B", "m", "eta", "pi", "a", "c", "chiQ", "chiH", "rhoI", "rhoQ", "rhoH", "thetaQ"]
P0 = np.array([3.0, 0.002, 0.5, 0.20, 0.70, 2.0, 0.15, 0.25, 0.788, 0.05, 0.20, 0.20], dtype=float)
I0 = 1.0e-6
Y0 = np.array([1.0 - I0, 0.0, I0, 0.0, 0.0, 0.0], dtype=float)


def unpack(p):
    return tuple(float(x) for x in p)


def R0_value(p=P0):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(p)
    return B * eta / (eta + m) * (1.0 + theta * pi / (a + chiQ))


def normal_equilibrium(p=P0):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(p)
    R0 = R0_value(p)
    KQ = pi / (a + chiQ)
    KH = a * pi / ((a + chiQ) * chiH)
    KR = (rhoI + rhoQ * KQ + rhoH * KH) / m
    G = 1.0 / eta + 1.0 + KQ + KH + KR
    L = (eta + m) / (m * eta)
    i = (R0 - 1.0) / (G + L * (R0 - 1.0))
    y = np.array([1.0 - L * i, i / eta, i, KQ * i, KH * i, KR * i])
    bcrit = y[4] + a / c * y[3]
    return y, bcrit


def capacity_equilibria(b, p=P0):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(p)
    L = (eta + m) / (m * eta)
    h = c / (c + chiH) * b
    A = c * chiH / (c + chiH) * b
    i_min = A * (a + chiQ) / (a * pi)
    i_max = 1.0 / L * (1.0 - 1e-12)

    def q(i):
        return (pi * i - A) / chiQ

    def s(i):
        return 1.0 - L * i

    def r(i):
        return (rhoI * i + rhoQ * q(i) + rhoH * h) / m

    def n(i):
        return s(i) + i / eta + i + q(i) + h + r(i)

    def fc(i):
        return B * s(i) * (i + theta * q(i)) - (eta + m) / eta * i * n(i)

    if i_min > i_max:
        return []
    xs = np.linspace(i_min, i_max, 2501)
    vals = np.array([fc(x) for x in xs])
    roots = []
    if abs(vals[0]) < 1e-12:
        roots.append(xs[0])
    for x1, x2, f1, f2 in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if f1 * f2 < 0.0:
            roots.append(brentq(fc, x1, x2, xtol=1e-14))
    out = []
    for i in sorted(set(round(float(x), 14) for x in roots)):
        out.append(np.array([s(i), i / eta, i, q(i), h, r(i)]))
    return out


def rhs_normal(t, y, p=P0):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(p)
    s, e, i, q, h, r = y
    n = np.sum(y)
    incidence = B * s * (i + theta * q) / n
    A = a * q
    return np.array([
        m * (1.0 - s) - incidence,
        incidence - (eta + m) * e,
        eta * e - i,
        pi * i - A - chiQ * q,
        A - chiH * h,
        rhoI * i + rhoQ * q + rhoH * h - m * r,
    ])


def rhs_full(t, y, b, p=P0):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(p)
    s, e, i, q, h, r = y
    n = np.sum(y)
    incidence = B * s * (i + theta * q) / n
    A = min(a * q, c * max(b - h, 0.0))
    return np.array([
        m * (1.0 - s) - incidence,
        incidence - (eta + m) * e,
        eta * e - i,
        pi * i - A - chiQ * q,
        A - chiH * h,
        rhoI * i + rhoQ * q + rhoH * h - m * r,
    ])


def jacobian(y, b, regime, p=P0):
    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(p)
    s, e, i, q, h, r = y
    n = np.sum(y)
    u = i + theta * q
    Ts = B * u * (n - s) / n**2
    Te = -B * s * u / n**2
    Ti = B * s * (n - u) / n**2
    Tq = B * s * (theta * n - u) / n**2
    Th = -B * s * u / n**2
    Tr = Th
    J = np.array([
        [-m - Ts, -Te, -Ti, -Tq, -Th, -Tr],
        [Ts, Te - (eta + m), Ti, Tq, Th, Tr],
        [0, eta, -1, 0, 0, 0],
        [0, 0, pi, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, rhoI, rhoQ, rhoH, -m],
    ], dtype=float)
    if regime == "N":
        J[3, 3] = -(a + chiQ)
        J[4, 3] = a
        J[4, 4] = -chiH
    elif regime == "C":
        J[3, 3] = -chiQ
        J[3, 4] = c
        J[4, 4] = -(c + chiH)
    else:
        raise ValueError("regime must be 'N' or 'C'")
    return J


def normal_trajectory(p=P0, t_end=500.0, rtol=1e-12, atol=1e-15):
    return solve_ivp(lambda t, y: rhs_normal(t, y, p), (0.0, t_end), Y0,
                     method="DOP853", rtol=rtol, atol=atol,
                     dense_output=True, max_step=0.1)


def transient_threshold(p=P0, t_end=160.0):
    a, c = p[4], p[5]
    sol = normal_trajectory(p, t_end=t_end, rtol=3e-10, atol=1e-13)
    ts = np.linspace(0.0, min(100.0, t_end), 1201)
    yy = sol.sol(ts)
    pp = yy[4] + a / c * yy[3]
    j = int(np.argmax(pp))
    lo = ts[max(0, j - 2)]
    hi = ts[min(len(ts) - 1, j + 2)]
    opt = minimize_scalar(lambda t: -(sol.sol(t)[4] + a / c * sol.sol(t)[3]),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    return -opt.fun, opt.x, sol


def first_lyapunov_coefficient(yH, bH):
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover
        return None, f"JAX unavailable: {exc}"
    jax.config.update("jax_enable_x64", True)

    p = tuple(float(x) for x in P0)

    def f(y):
        B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = p
        s, e, i, q, h, r = y
        n = jnp.sum(y)
        incidence = B * s * (i + theta * q) / n
        A = c * (bH - h)
        return jnp.array([
            m * (1.0 - s) - incidence,
            incidence - (eta + m) * e,
            eta * e - i,
            pi * i - A - chiQ * q,
            A - chiH * h,
            rhoI * i + rhoQ * q + rhoH * h - m * r,
        ])

    yj = jnp.array(yH)
    A = np.array(jax.jacfwd(f)(yj))
    H = np.array(jax.jacfwd(jax.jacfwd(f))(yj))
    C3 = np.array(jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(yj))
    vals, vl, vr = eig(A, left=True, right=True)
    k = int(np.argmax(vals.imag))
    lam = vals[k]
    q = vr[:, k]
    pvec = vl[:, k]
    pvec = pvec / np.conj(np.vdot(pvec, q))
    omega = lam.imag

    def Bform(x, y):
        return np.einsum("ijk,j,k->i", H, x, y)

    def Cform(x, y, z):
        return np.einsum("ijkl,j,k,l->i", C3, x, y, z)

    g21 = np.vdot(
        pvec,
        Cform(q, q, np.conj(q))
        - 2.0 * Bform(q, np.linalg.solve(A, Bform(q, np.conj(q))))
        + Bform(np.conj(q), np.linalg.solve(2j * omega * np.eye(6) - A, Bform(q, q))),
    )
    return float(g21.real / (2.0 * omega)), None


def find_hopf(bcrit):
    def alpha_c(b):
        roots = capacity_equilibria(b)
        if not roots:
            return np.nan
        return float(np.max(eigvals(jacobian(roots[0], b, "C")).real))

    bs = np.geomspace(1e-7, bcrit * (1.0 - 1e-9), 600)
    av = np.array([alpha_c(b) for b in bs])
    idx = np.where(av[:-1] * av[1:] < 0.0)[0]
    if len(idx) != 1:
        raise RuntimeError(f"Expected one Hopf crossing, found {len(idx)}")
    j = idx[0]
    bH = brentq(alpha_c, bs[j], bs[j + 1], xtol=1e-15)
    yH = capacity_equilibria(bH)[0]
    ev = eigvals(jacobian(yH, bH, "C"))
    db = 1e-8
    trans = (alpha_c(bH + db) - alpha_c(bH - db)) / (2.0 * db)
    return bH, yH, ev, float(trans)


def simulate_cycle(b, bcrit):
    eq = capacity_equilibria(b)[0]
    y0 = eq.copy()
    y0[2] *= 1.0 + 1e-6
    sol = solve_ivp(lambda t, y: rhs_full(t, y, b), (0.0, 10000.0), y0,
                    method="Radau", rtol=1e-10, atol=1e-13,
                    max_step=1.0, dense_output=True)
    mask = sol.t > 7000.0
    peaks, _ = find_peaks(sol.y[3, mask], distance=100)
    tp = sol.t[mask][peaks]
    period = float(np.mean(np.diff(tp[-8:])))
    t1, t2 = tp[-2], tp[-1]
    tt = np.linspace(t1, t2, 20001)
    yy = sol.sol(tt)
    a, c = P0[4], P0[5]
    psi = a * yy[3] - c * (b - yy[4])
    fraction = float(np.mean(psi >= 0.0))
    return eq, period, fraction, tt, yy, psi


def first_root(func, level, t_end=100.0):
    ts = np.linspace(0.0, t_end, 3001)
    vals = np.array([func(t) - level for t in ts])
    idx = np.where(vals[:-1] * vals[1:] <= 0.0)[0]
    for j in idx:
        if ts[j + 1] <= 1e-10:
            continue
        if vals[j] == 0.0 and ts[j] == 0.0:
            continue
        return brentq(lambda t: func(t) - level, ts[j], ts[j + 1], xtol=1e-13)
    return np.nan


def make_outputs():
    R0 = R0_value()
    yN, bcrit = normal_equilibrium()
    btrans, tstar, solN = transient_threshold(P0, 500.0)
    a, c = P0[4], P0[5]

    def pressure(t):
        y = solN.sol(t)
        return float(y[4] + a / c * y[3])

    def pressure_prime(t):
        y = solN.sol(t)
        dy = rhs_normal(t, y)
        return float(dy[4] + a / c * dy[3])

    hh = 1e-4
    curvature = -(pressure_prime(tstar + hh) - pressure_prime(tstar - hh)) / (2.0 * hh)

    JN = jacobian(yN, bcrit, "N")
    JC = jacobian(yN, bcrit, "C")
    bH, yH, evH, trans = find_hopf(bcrit)
    l1, l1_error = first_lyapunov_coefficient(yH, bH)

    bcycle = 0.5 * bcrit
    eq_cycle, period, frac_cap, tcycle, ycycle, psi_cycle = simulate_cycle(bcycle, bcrit)

    bs = np.geomspace(5e-5, bcrit * (1.0 - 1e-6), 220)
    alpha_c = []
    for b in bs:
        roots = capacity_equilibria(b)
        alpha_c.append(np.max(eigvals(jacobian(roots[0], b, "C")).real) if roots else np.nan)
    alpha_n = np.max(eigvals(JN).real)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].semilogx(bs, alpha_c, label=r"capacity-limited ($J_C$)")
    ax[0].axhline(alpha_n, label=r"normal-care ($J_N$)")
    ax[0].axhline(0.0, linestyle="--", linewidth=0.8)
    ax[0].axvline(bH, linestyle="--", linewidth=0.9, label=r"$b_H$")
    ax[0].axvline(bcrit, linestyle=":", linewidth=0.9, label=r"$b_{\rm crit}$")
    ax[0].set_xlabel("hospital capacity $b$")
    ax[0].set_ylabel("spectral abscissa")
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) one-sided stability")
    ax[1].plot(ycycle[3], ycycle[4], label="attracting cycle")
    qline = np.linspace(0.0, max(ycycle[3]) * 1.03, 300)
    hline = bcycle - a / c * qline
    valid = hline >= 0.0
    ax[1].plot(qline[valid], hline[valid], linestyle="--", label=r"$\Sigma$")
    ax[1].plot(eq_cycle[3], eq_cycle[4], marker="x", linestyle="none", label="unstable equilibrium")
    ax[1].set_xlabel("severe waiting class $q$")
    ax[1].set_ylabel("hospital occupancy $h$")
    ax[1].set_title(r"(b) observed cycle at $b=0.5b_{\rm crit}$")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    eps = np.logspace(-7, -4, 7)
    durations, deficits = [], []
    for ep in eps:
        b = btrans - ep
        sol = solve_ivp(lambda t, y: rhs_full(t, y, b), (0.0, 80.0), Y0,
                        method="DOP853", rtol=1e-12, atol=1e-15,
                        dense_output=True, max_step=0.01)
        grid = np.linspace(tstar - 2.0, tstar + 2.0, 20001)
        psi = a * sol.sol(grid)[3] - c * (b - sol.sol(grid)[4])
        ids = np.where(psi[:-1] * psi[1:] < 0.0)[0]
        roots = [brentq(lambda t: a * sol.sol(t)[3] - c * (b - sol.sol(t)[4]),
                        grid[j], grid[j + 1], xtol=1e-13) for j in ids]
        t1, t2 = roots[0], roots[-1]
        durations.append(t2 - t1)
        deficits.append(quad(lambda t: max(a * sol.sol(t)[3] - c * (b - sol.sol(t)[4]), 0.0),
                             t1, t2, epsabs=1e-18, epsrel=1e-10, limit=200)[0])
    durations = np.array(durations)
    deficits = np.array(deficits)
    duration_const = 2.0 * np.sqrt(2.0 / curvature)
    deficit_const = 4.0 * np.sqrt(2.0) * c / (3.0 * np.sqrt(curvature))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].semilogx(eps, durations / np.sqrt(eps), marker="o", label="numerical")
    ax[0].axhline(duration_const, linestyle="--", label="asymptotic constant")
    ax[0].set_xlabel(r"$\varepsilon=b_{\rm trans}-b$")
    ax[0].set_ylabel(r"$\Delta\tau/\varepsilon^{1/2}$")
    ax[0].set_title("(a) duration compensation")
    ax[0].legend(fontsize=8)
    ax[1].semilogx(eps, deficits / eps**1.5, marker="o", label="numerical")
    ax[1].axhline(deficit_const, linestyle="--", label="asymptotic constant")
    ax[1].set_xlabel(r"$\varepsilon=b_{\rm trans}-b$")
    ax[1].set_ylabel(r"$D_{\rm adm}/\varepsilon^{3/2}$")
    ax[1].set_title("(b) admission-deficit compensation")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grazing.pdf", bbox_inches="tight")
    plt.close(fig)

    B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = unpack(P0)
    F = np.array([[0.0, B, B * theta, 0.0], [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    V = np.array([[eta + m, 0.0, 0.0, 0.0], [-eta, 1.0, 0.0, 0.0],
                  [0.0, -pi, a + chiQ, 0.0], [0.0, 0.0, -a, chiH]])
    M = F - V
    w = np.array([0.0, 0.0, a / c, 1.0])
    X0 = np.array([0.0, I0, 0.0, 0.0])

    def pplus(t):
        return float(w @ expm(M * t) @ X0)

    bvals = np.geomspace(1.05 * bcrit, 0.995 * btrans, 45)
    true_t = np.array([first_root(pressure, b) for b in bvals])
    lower_t = np.array([first_root(pplus, b) for b in bvals])
    relerr = (true_t - lower_t) / true_t
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.semilogx(bvals, true_t, label=r"true $\tau_c$")
    ax.semilogx(bvals, lower_t, linestyle="--", label=r"lower bound $\tau_c^+$")
    ax.set_xlabel("hospital capacity $b$")
    ax.set_ylabel("first switching time")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.semilogx(bvals, 100.0 * relerr, linestyle=":", label="relative gap")
    ax2.set_ylabel("bound gap (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_switchtime.pdf", bbox_inches="tight")
    plt.close(fig)

    def thresholds(p):
        R = R0_value(p)
        y, bc = normal_equilibrium(p)
        bt, _, _ = transient_threshold(p, 160.0)
        return np.array([R, bc, bt])

    hrel = 1e-4
    E = np.zeros((3, len(P0)))
    for j, x in enumerate(P0):
        pp, pm = P0.copy(), P0.copy()
        pp[j] = x * (1.0 + hrel)
        pm[j] = x * (1.0 - hrel)
        E[:, j] = (np.log(thresholds(pp)) - np.log(thresholds(pm))) / (
            np.log(1.0 + hrel) - np.log(1.0 - hrel)
        )
    x = np.arange(len(NAMES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    labels = [r"$R_0$", r"$b_{\rm crit}$", r"$b_{\rm trans}$"]
    for k in range(3):
        ax.bar(x + (k - 1) * width, E[k], width, label=labels[k])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\mathcal{B}$", r"$m$", r"$\eta$", r"$\pi$", r"$a$", r"$c$",
                        r"$\chi_Q$", r"$\chi_H$", r"$\rho_I$", r"$\rho_Q$", r"$\rho_H$", r"$\theta_Q$"])
    ax.set_ylabel(r"elasticity $\partial\ln Y/\partial\ln X$")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    Bvals = np.linspace(0.6, 6.0, 65)
    bcvals = np.full_like(Bvals, np.nan)
    btvals = np.full_like(Bvals, np.nan)
    Rvals = np.empty_like(Bvals)
    for j, Bb in enumerate(Bvals):
        p = P0.copy(); p[0] = Bb
        Rvals[j] = R0_value(p)
        if Rvals[j] > 1.0:
            _, bcvals[j] = normal_equilibrium(p)
            btvals[j], _, _ = transient_threshold(p, 160.0)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(Bvals, bcvals, label=r"$b=b_{\rm crit}$")
    ax.plot(Bvals, btvals, label=r"$b=b_{\rm trans}$")
    ax.fill_between(Bvals, bcvals, btvals, where=np.isfinite(btvals), alpha=0.25,
                    label="transient-overload band")
    B_inv = (P0[2] + P0[1]) / P0[2] / (1.0 + P0[11] * P0[3] / (P0[4] + P0[6]))
    ax.axvline(B_inv, linestyle="--", linewidth=0.9, label=r"$R_0=1$")
    ax.set_xlabel(r"transmission parameter $\mathcal{B}$")
    ax.set_ylabel("hospital capacity $b$")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regime.pdf", bbox_inches="tight")
    plt.close(fig)

    results = {
        "R0": R0,
        "bcrit": bcrit,
        "btrans": btrans,
        "btrans_over_bcrit": btrans / bcrit,
        "tau_star": tstar,
        "curvature_k": curvature,
        "JN_eigenvalues": [[float(z.real), float(z.imag)] for z in eigvals(JN)],
        "JC_boundary_eigenvalues": [[float(z.real), float(z.imag)] for z in eigvals(JC)],
        "det_JN": float(det(JN)),
        "det_JC": float(det(JC)),
        "bH": bH,
        "bH_over_bcrit": bH / bcrit,
        "hopf_eigenvalues": [[float(z.real), float(z.imag)] for z in evH],
        "hopf_transversality": trans,
        "first_lyapunov_coefficient": l1,
        "first_lyapunov_error": l1_error,
        "hopf_criticality": "subcritical" if (l1 is not None and l1 > 0) else "undetermined",
        "observed_cycle_b": bcycle,
        "observed_cycle_period": period,
        "observed_cycle_capacity_fraction": frac_cap,
        "duration_asymptotic_constant": duration_const,
        "deficit_asymptotic_constant": deficit_const,
        "switching_lower_bound_max_relative_gap": float(np.nanmax(relerr)),
        "elasticities": {label: {name: float(E[k, j]) for j, name in enumerate(NAMES)}
                         for k, label in enumerate(["R0", "bcrit", "btrans"])},
    }
    with open(OUT / "verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    make_outputs()
