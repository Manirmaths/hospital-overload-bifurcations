#!/usr/bin/env python3
"""R2.11: elasticities of R0, b_crit, b_trans with respect to the independent
DIMENSIONAL rates, propagating every algebraic dependency through the scaling
of Appendix A (kappa_I = p*alpha + delta + mu + d_I, chi's are total removal
rates, pi + rho_I + m + Delta_I = 1 holds automatically).

Baseline: time measured in units of 1/kappa_I, so dimensional rates equal the
baseline dimensionless groups (kappa_I = 1).  Because every threshold is a
function of rate *ratios*, elasticities are independent of the time unit and
satisfy the Euler identity  sum_{rates} E = 0  (checked below).
Capacities are per N_* = Lambda/mu; with Lambda fixed, the elasticity of the
absolute bed requirement B_* = b_* N_* to mu is E(b_*, mu) - 1.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_revision import R0_value, normal_equilibrium
from thresholds import transient_threshold_certified, check_threshold_ordering

OUT = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parent

DIM = dict(beta=3.0, mu=0.002, sigma=0.5, palpha=0.2, delta=0.788, dI=0.010, nu=0.7, kappa=2.0,
           gammaQ=0.05, dQ=0.098, gammaH=0.2, dH=0.048, thetaQ=0.2)
LABEL = dict(beta=r"$\beta$", mu=r"$\mu$", sigma=r"$\sigma$", palpha=r"$p\alpha$", delta=r"$\delta$",
             dI=r"$d_I$", nu=r"$\nu$", kappa=r"$\kappa$", gammaQ=r"$\gamma_Q$", dQ=r"$d_Q$",
             gammaH=r"$\gamma_H$", dH=r"$d_H$", thetaQ=r"$\theta_Q$")


def to_dimless(d):
    kI = d["palpha"] + d["delta"] + d["mu"] + d["dI"]
    return np.array([d["beta"] / kI, d["mu"] / kI, d["sigma"] / kI, d["palpha"] / kI,
                     d["nu"] / kI, d["kappa"] / kI,
                     (d["gammaQ"] + d["dQ"] + d["mu"]) / kI, (d["gammaH"] + d["dH"] + d["mu"]) / kI,
                     d["delta"] / kI, d["gammaQ"] / kI, d["gammaH"] / kI, d["thetaQ"]])


def thresholds(d):
    p = to_dimless(d)
    r = transient_threshold_certified(p)
    check_threshold_ordering(r)
    return np.array([R0_value(p), r.bcrit, r.btrans])


def main(h=1e-4):
    from verify_revision import P0
    assert np.allclose(to_dimless(DIM), P0), "baseline mapping mismatch"
    names = list(DIM)
    E = np.zeros((3, len(names)))
    for j, k in enumerate(names):
        dp, dm = dict(DIM), dict(DIM)
        dp[k] *= 1 + h; dm[k] *= 1 - h
        E[:, j] = (np.log(thresholds(dp)) - np.log(thresholds(dm))) / (np.log(1 + h) - np.log(1 - h))
    rates = [k for k in names if k != "thetaQ"]
    euler = {lab: float(sum(E[i, names.index(k)] for k in rates)) for i, lab in enumerate(["R0", "bcrit", "btrans"])}
    res = dict(h=h, elasticities={lab: dict(zip(names, map(float, E[i])))
                                  for i, lab in enumerate(["R0", "bcrit", "btrans"])},
               euler_sum_over_rates=euler)
    json.dump(res, open(OUT / "dimensional_sensitivity.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    for k, lab in enumerate([r"$R_0$", r"$b_{\rm crit}$", r"$b_{\rm trans}$"]):
        ax.bar(x + (k - 1) * w, E[k], w, label=lab)
    ax.axhline(0, lw=0.8, color="k")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[k] for k in names])
    ax.set_ylabel(r"elasticity $\partial\ln Y/\partial\ln X$")
    ax.set_title("(b) independent dimensional rates (all dependencies propagated)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "fig_sensitivity_dimensional.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
