#!/usr/bin/env python3
"""Figures added or regenerated for revision 1.1.0.

  fig_global_bifurcation.pdf  shooting branches + collocation overlay + grazing point
  fig_bistability.pdf         R2.3: trajectories and basin slices
  fig_grazing.pdf             R2.5: compensated laws + convergence diagnostics
  fig_sensitivity.pdf         R2.11: formal dimensionless + dimensional elasticities
  fig_bifurcation_atlas.pdf   R2.1/R2.8/R2.13: corrected atlas, log scale, fold curves
  fig_switchtime.pdf          R1.3: bound gap versus b/b_trans

Usage: python make_revision_figures.py [names...]   (default: all available)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7.5})


def load(n):
    return json.load(open(R / n))


def fig_global():
    def ld(n):
        return json.load(open(ROOT / n))
    uh = ld("unstable_branch_hopf.json") + ld("unstable_branch_hopf_tail.json")
    sd, su = ld("stable_branch_down.json"), ld("stable_branch_up.json")
    qm = []
    for p in sorted((ROOT / "qmax_branch_parts").glob("qmax_branch_*.json")):
        qm += json.load(open(p))
    cl = load("collocation_branch_tol1e-06_dir-1.json")["rows"]
    cu = load("collocation_branch_tol1e-06_dir1.json")["rows"]
    fl = load("collocation_fold_refine_lower.json"); fu = load("collocation_fold_refine_upper.json")
    bL = fl[-1]["b_fold"]; bU = fu[-1]["b_fold"]
    bH = 2.32155208e-4; bc = 1.0256571258926393e-3
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axs[0]
    for data, st, lab, col in [(uh, "--", "unstable cycles (shooting)", "C3"), (sd, "-", "stable cycles (shooting)", "C0"),
                               (su, "-", None, "C0"), (qm, "--", None, "C3")]:
        ax.plot([x["b"] for x in data], [x["ampq"] for x in data], st, lw=1.6, color=col, label=lab)
    for rows, lab in [(cl, "collocation (independent)"), (cu, None)]:
        ax.plot([x["b"] for x in rows], [x["ampq"] for x in rows], "o", ms=2.6, mfc="none", color="k", mew=0.6, label=lab)
    ax.plot([4e-5, bH], [0, 0], "-", lw=2, color="C2", label="stable equilibrium")
    ax.plot([bH, bc], [0, 0], "--", lw=2, color="C1", label="unstable equilibrium")
    ax.plot([bc, 2.4e-3], [0, 0], "-", lw=2, color="C2")
    ax.plot(bH, 0, "^", ms=7, color="k", label="Hopf $H$")
    ax.plot([bL, bU], [fl[-1]["fold_row"]["ampq"], fu[-1]["fold_row"]["ampq"]], "s", ms=6, color="C4", label="folds $LPC^\\pm$")
    ax.plot(1.0074e-4, 7.94e-3, "D", ms=5, color="C5", label=r"grazing of unstable branch $b_g$")
    ax.axvspan(bL, bH, color="C2", alpha=0.08); ax.axvspan(bc, bU, color="C2", alpha=0.08)
    ax.axvline(bc, ls=":", lw=1)
    ax.text(bc * 1.02, 0.0135, r"$b_{\rm crit}$", rotation=90, va="top", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(5e-5, 2.4e-3); ax.set_ylim(-4.5e-4, 0.0142)
    ax.set_xlabel("hospital capacity $b$"); ax.set_ylabel(r"cycle amplitude $\max q-\min q$")
    ax.set_title("(a) periodic-orbit branches; shaded: bistable windows")
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)
    ax = axs[1]
    for data, st, lab in [(uh, "--", "unstable branch"), (sd, "-", "stable branch"), (su, "-", None), (qm, "--", None)]:
        xs = [x["b"] for x in data if x.get("rho")]; ys = [x["rho"] for x in data if x.get("rho")]
        ax.plot(xs, ys, st, lw=1.6, label=lab)
    ax.axhline(1, ls=":", lw=1)
    for v in (bH, bL, bU): ax.axvline(v, ls=":", lw=1)
    ax.axvline(1.0074e-4, ls="-.", lw=0.9, color="C5")
    ax.text(1.03e-4, 4.5, "$b_g$", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e-5, 2.4e-3); ax.set_ylim(0.45, 6)
    ax.set_xlabel("hospital capacity $b$"); ax.set_ylabel("largest nontrivial Floquet multiplier modulus")
    ax.set_title("(b) cycle stability (shooting)"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "fig_global_bifurcation.pdf", bbox_inches="tight"); plt.close(fig)


def fig_bistab():
    d = load("bistability.json")
    fig, ax = plt.subplots(2, 2, figsize=(10.4, 7.2))
    for col, key in enumerate(["0.00015", "0.0015"]):
        r = d[key]; b = r["b"]
        t = np.array(r["traj_t"])
        a = ax[0, col]
        a.plot(t, r["traj_eq_q"], lw=0.9, label="start near equilibrium")
        a.plot(t, r["traj_cyc_q"], lw=0.9, label="start on large cycle")
        a.set_xlabel(r"time $\tau$"); a.set_ylabel("waiting class $q$")
        reg = r"$(b_{\rm LPC}^-,b_H)$" if b < 1e-3 else r"$(b_{\rm crit},b_{\rm LPC}^+)$"
        bl = r"1.5\times10^{-4}" if b < 1e-3 else r"1.5\times10^{-3}"
        a.set_title(f"({'ab'[col]}) $b={bl}$, inside {reg}")
        a.legend(loc="upper right")
        a = ax[1, col]
        q, h = np.array(r["slice_q"]), np.array(r["slice_h"])
        amp = np.array(r["slice_amp"])
        Z = (amp > 1e-4).astype(float)
        a.pcolormesh(q, h, Z, cmap=ListedColormap(["#d9ead3", "#f4cccc"]), shading="nearest", vmin=0, vmax=1)
        a.plot(r["traj_cyc_q"][-600:], r["traj_cyc_h"][-600:], color="C1", lw=1.0, label="attracting cycle")
        E = r["equilibrium"]
        a.plot(E[3], E[4], "k*", ms=9, label="stable equilibrium")
        a2 = 1.0 if b < 1e-3 else 1.0
        qq = np.linspace(0, q.max(), 50)
        from verify_revision import P0
        a_, c_ = P0[4], P0[5]
        hh = b - a_ / c_ * qq
        ok = hh >= 0
        a.plot(qq[ok], hh[ok], "k--", lw=0.8, label=r"$\Sigma$")
        a.set_xlabel("initial $q$"); a.set_ylabel("initial $h$")
        a.set_title(f"({'cd'[col]}) basin slice: green $\\to$ equilibrium, red $\\to$ cycle")
        a.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(); fig.savefig(ROOT / "fig_bistability.pdf", bbox_inches="tight"); plt.close(fig)


def fig_grazing():
    d = load("grazing_diagnostics.json")
    Kd, Kf, k = d["duration_constant"], d["deficit_constant"], d["k"]
    from verify_revision import P0
    a_, c_ = P0[4], P0[5]
    fig, ax = plt.subplots(2, 2, figsize=(10.0, 6.8))
    ax = ax.ravel()
    for key, mk in [("1e-10", "o"), ("1e-12", "x")]:
        rows = d["runs"][key]
        e = np.array([r["eps"] for r in rows])
        floor = 1e-7 if key == "1e-12" else 1e-6     # below: integration-error floor
        ok = e >= floor
        ax[0].semilogx(e, [r["dur_comp"] for r in rows], mk, ms=4, mfc="none", label=f"rtol {key}")
        ax[1].semilogx(e, [r["def_comp"] for r in rows], mk, ms=4, mfc="none", label=f"rtol {key}")
        ax[2].semilogx(e[ok], np.array([r["exit_shift_over_eps"] for r in rows])[ok], mk, ms=4, mfc="none",
                       label=f"rtol {key}")
        ax[3].semilogx(e[ok], np.array([r["dev200_comp"] for r in rows])[ok], mk, ms=4, mfc="none",
                       label=r"$\tau\in[0,200]$, rtol " + key)
        ax[3].semilogx(e[ok], np.array([r["dev2000_comp"] for r in rows])[ok], mk, ms=3,
                       label=r"$\tau\in[0,2000]$, rtol " + key)
    ax[0].axhline(Kd, ls="--", color="k", lw=0.8, label=r"$2\sqrt{2/k}$")
    ax[1].axhline(Kf, ls="--", color="k", lw=0.8, label=r"$(4\sqrt{2}/3)\,c\,k^{-1/2}$")
    ax[2].axhline(4 * (a_ - c_) / (3 * k), ls="--", color="k", lw=0.8, label=r"$4(a-c)/(3k)$")
    titles = ["(a) overload duration", "(b) cumulative admission deficit",
              "(c) exit-time shift caused by the perturbation", "(d) trajectory deviation"]
    ylabs = [r"$\Delta\tau/\varepsilon^{1/2}$", r"$D_{\rm adm}/\varepsilon^{3/2}$",
             r"$(\tau_2-\tau_2^N)/\varepsilon$", r"$\sup_\tau\|y-y_N\|_\infty/\varepsilon^{3/2}$"]
    for a, t, yl in zip(ax, titles, ylabs):
        a.set_xlabel(r"$\varepsilon=b_{\rm trans}-b$"); a.set_title(t); a.set_ylabel(yl); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "fig_grazing.pdf", bbox_inches="tight"); plt.close(fig)


def fig_sens():
    import verify_revision as vr
    from thresholds import transient_threshold_certified
    names = vr.NAMES
    def thr(p):
        r = transient_threshold_certified(p)
        return np.array([vr.R0_value(p), r.bcrit, r.btrans])
    h = 1e-4
    E = np.zeros((3, len(names)))
    for j, x in enumerate(vr.P0):
        pp, pm = vr.P0.copy(), vr.P0.copy(); pp[j] = x * (1 + h); pm[j] = x * (1 - h)
        E[:, j] = (np.log(thr(pp)) - np.log(thr(pm))) / (np.log(1 + h) - np.log(1 - h))
    json.dump({lab: dict(zip(names, map(float, E[i]))) for i, lab in enumerate(["R0", "bcrit", "btrans"])},
              open(R / "formal_dimensionless_sensitivity.json", "w"), indent=1)
    dd = load("dimensional_sensitivity.json")["elasticities"]
    from dimensional_sensitivity import LABEL
    dn = list(dd["R0"].keys())
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 7.2))
    lab = [r"$R_0$", r"$b_{\rm crit}$", r"$b_{\rm trans}$"]
    x = np.arange(len(names)); w = 0.26
    for k in range(3):
        ax[0].bar(x + (k - 1) * w, E[k], w, label=lab[k])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r"$\mathcal{B}$", r"$m$", r"$\eta$", r"$\pi$", r"$a$", r"$c$", r"$\chi_Q$", r"$\chi_H$",
                           r"$\rho_I$", r"$\rho_Q$", r"$\rho_H$", r"$\theta_Q$"])
    ax[0].set_title("(a) formal sensitivities to dimensionless groups (others held fixed; not physically realisable)")
    x = np.arange(len(dn))
    for k, key in enumerate(["R0", "bcrit", "btrans"]):
        ax[1].bar(x + (k - 1) * w, [dd[key][n] for n in dn], w, label=lab[k])
    ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[n] for n in dn])
    ax[1].set_title("(b) operational sensitivities to independent dimensional rates (all dependencies propagated)")
    for a in ax:
        a.axhline(0, lw=0.8, color="k"); a.set_ylabel(r"elasticity $\partial\ln Y/\partial\ln X$"); a.legend(ncol=3)
    fig.tight_layout(); fig.savefig(ROOT / "fig_sensitivity.pdf", bbox_inches="tight"); plt.close(fig)


def fig_atlas():
    d = load("threshold_atlas.json"); rows = d["rows"]
    B = np.array([r["B"] for r in rows]); bc = np.array([r["bcrit"] for r in rows])
    bt = np.array([r["btrans"] for r in rows]); old = np.array([r["btrans_v100"] for r in rows])
    bH = np.array([r["bH"][0] if r["bH"] else np.nan for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6))
    a = ax[0]
    a.plot(B, bt, lw=1.8, label=r"$b_{\rm trans}$ (certified)")
    a.plot(B, old, ":", lw=1.2, color="0.4", label=r"$b_{\rm trans}$, v1.0 search on $\tau\leq100$ (erroneous)")
    a.plot(B, bc, lw=1.8, label=r"$b_{\rm crit}$")
    a.plot(B, bH, "--", lw=1.6, label=r"$b_H$ (subcritical Hopf)")
    try:
        rb = load("robustness_Bcurve.json") + [x for x in load("robustness_oat.json") if x["label"] == "baseline"]
        rb = sorted([x for x in rb if x.get("fold_lower", {}).get("estimate")], key=lambda x: x["params"]["B"])
        a.plot([x["params"]["B"] for x in rb], [x["fold_lower"]["estimate"] for x in rb], "v-", ms=4, lw=1,
               color="C4", label=r"$b_{\rm LPC}^-$ (fold, simulation bracket)")
        a.plot([x["params"]["B"] for x in rb], [x["fold_upper"]["estimate"] for x in rb], "^-", ms=4, lw=1,
               color="C5", label=r"$b_{\rm LPC}^+$ (fold, simulation bracket)")
    except FileNotFoundError:
        pass
    a.fill_between(B, bc, bt, alpha=0.15, label="transient-overload band")
    a.fill_between(B, bH, bc, alpha=0.15, label="unstable capacity equilibrium")
    a.axvline(d["B_invasion"], ls=":", lw=1, color="k", label=r"$R_0=1$")
    a.set_yscale("log"); a.set_ylim(3e-6, 0.2); a.set_xlim(0.9, 6.05)
    a.set_xlabel(r"transmission parameter $\mathcal{B}$"); a.set_ylabel("hospital capacity $b$ (log scale)")
    a.set_title("(a) threshold and stability atlas"); a.legend(fontsize=6.3, loc="lower right", ncol=2)
    a = ax[1]
    a.plot(B, bt / bc, lw=1.8, color="k", label="baseline seed $i(0)=10^{-6}$")
    ic = load("ic_robustness.json")
    for name, mk in [("i0=1e-08", "o"), ("i0=0.0001", "s"), ("i0=0.001", "d"), ("e0=1e-06", "x"),
                     ("immune=0.1,i0=1e-6", "^"), ("immune=0.3,i0=1e-6", "v")]:
        a.plot([x["B"] for x in ic[name]], [x["ratio"] for x in ic[name]], mk, ms=5, mfc="none", label=name)
    a.axhline(1, ls=":", color="k", lw=0.8)
    a.axvline(d["B_invasion"], ls=":", lw=1, color="k")
    a.set_xlabel(r"transmission parameter $\mathcal{B}$"); a.set_ylabel(r"$b_{\rm trans}/b_{\rm crit}$")
    a.set_title("(b) threshold separation and initial-condition dependence"); a.legend(fontsize=6.5)
    a.set_xlim(0.9, 6.05)
    fig.tight_layout(); fig.savefig(ROOT / "fig_bifurcation_atlas.pdf", bbox_inches="tight"); plt.close(fig)


def fig_switch():
    import verify_revision as vr
    from scipy.linalg import expm
    from thresholds import transient_threshold_certified
    r = transient_threshold_certified(vr.P0); sol = r.sol; bt = r.btrans
    B_, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, vr.P0)
    F = np.array([[0, B_, B_ * theta, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.]])
    V = np.array([[eta + m, 0, 0, 0], [-eta, 1, 0, 0], [0, -pi, a + chiQ, 0], [0, 0, -a, chiH]])
    M = F - V; w = np.array([0, 0, a / c, 1.0]); X0 = np.array([0, vr.I0, 0, 0])
    P = lambda t: float(sol.sol(t)[4] + a / c * sol.sol(t)[3])
    Pp = lambda t: float(w @ expm(M * t) @ X0)
    bvals = np.geomspace(1.05 * r.bcrit, 0.999 * bt, 60)
    tt = np.array([vr.first_root(P, b) for b in bvals]); tl = np.array([vr.first_root(Pp, b) for b in bvals])
    sn = np.array([sol.sol(t)[0] / sol.sol(t).sum() for t in tt])
    gap = (tt - tl) / tt
    json.dump(dict(b=bvals.tolist(), b_over_btrans=(bvals / bt).tolist(), tau_true=tt.tolist(), tau_bound=tl.tolist(),
                   rel_gap=gap.tolist(), s_over_n_at_tau=sn.tolist()), open(R / "switchtime.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax[0].semilogx(bvals, tt, label=r"true $\tau_c$"); ax[0].semilogx(bvals, tl, "--", label=r"lower bound $\tau_c^+$")
    ax[0].set_xlabel("hospital capacity $b$"); ax[0].set_ylabel("first switching time"); ax[0].legend()
    ax[0].set_title("(a) first switching time and rigorous bound")
    ax[1].plot(bvals / bt, 100 * gap, label=r"relative gap $(\tau_c-\tau_c^+)/\tau_c$ (%)")
    ax[1].plot(bvals / bt, 100 * (1 - sn), ":", label=r"susceptible depletion $1-s/n$ at $\tau_c$ (%)")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$b/b_{\rm trans}$"); ax[1].set_ylabel("%")
    ax[1].set_title("(b) gap grows with susceptible depletion"); ax[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "fig_switchtime.pdf", bbox_inches="tight"); plt.close(fig)
    print("gap at 1.2 bcrit", np.interp(1.2 * r.bcrit, bvals, gap), "at 0.95 bt", np.interp(0.95 * bt, bvals, gap),
          "max", gap.max(), "s/n at max", sn[-1])


if __name__ == "__main__":
    todo = sys.argv[1:] or ["global", "bistab", "grazing", "sens", "atlas", "switch"]
    for t in todo:
        globals()["fig_" + t]()
        print("made", t)
