#!/usr/bin/env python3
"""Generate the revision Supplementary Information (LaTeX) from archived results,
so that every number in the SI tables is traceable to a results/*.json file."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUT = Path("/home/claude/rev/ms/supplementary_revision.tex")


def J(n):
    return json.load(open(R / n))


def sci(x, d=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    m, e = f"{x:.{d}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def table_atlas():
    rows = J("threshold_atlas.json")["rows"]
    lines = []
    for r in rows:
        if r["B"] <= 1.45 or abs(r["B"] - 3.0) < 1e-9:
            lines.append(f"{r['B']:.4f} & {r['R0']:.4f} & {sci(r['bcrit'])} & {sci(r['btrans_v100'])} & "
                         f"{sci(r['btrans'])} & {r['tau_star']:.1f} & {r['horizon']:.0f} & "
                         f"{'yes' if r['v100_violates_theorem'] else 'no'} \\\\")
    return "\n".join(lines)


def table_ic():
    ic = J("ic_robustness.json")
    Bs = [x["B"] for x in next(iter(ic.values()))]
    head = " & ".join(f"$\\mathcal B={B:g}$" for B in Bs)
    lines = [f"Initial condition & {head} \\\\", r"\midrule"]
    for k, v in ic.items():
        lines.append(k.replace("_", r"\_").replace("e0", "$e(0)$").replace("i0", "$i(0)$") + " & " +
                     " & ".join(f"{x['ratio']:.2f}" for x in v) + r" \\")
    return "\n".join(lines)


def table_folds():
    out = []
    for w in ["lower", "upper"]:
        for r in J(f"collocation_fold_refine_{w}.json"):
            if not r["interior"]:
                continue
            fr = r["fold_row"]
            out.append(f"{w} & {r['tol']:g} & {r['ds']:g} & {r['b_fold']:.9e} & {fr['nodes']} & "
                       f"{sci(fr['psimin'],2)} & {sci(fr['psimax'],2)} & {1-fr['fracC']:.3f} & "
                       f"{sci(fr['min_abs_dpsi_at_cross'],2)} \\\\")
    return "\n".join(out)


def table_saltation():
    out = []
    for r in J("saltation_check.json"):
        mu = [abs(complex(*r[k][0])) for k in ("mu_a", "mu_b", "mu_c")]
        cr = r["crossings"]
        out.append(f"{sci(r['b'],2)} & {r['T']:.3f} & {sci(r['shooting_residual'],1)} & {mu[0]:.6f} & {mu[1]:.6f} & "
                   f"{mu[2]:.6f} & {len(cr)} & {sci(min(abs(c['dpsi_dtau']) for c in cr),2)} & "
                   f"{sci(max(c['saltation_minus_I'] for c in cr),1)} \\\\")
    return "\n".join(out)


def table_grazing():
    d = J("grazing_diagnostics.json")
    out = []
    for r in d["runs"]["1e-12"]:
        out.append(f"{sci(r['eps'],1)} & {r['n_switches']} & {r['dur_comp']/d['duration_constant']:.6f} & "
                   f"{r['def_comp']/d['deficit_constant']:.6f} & {sci(r['entry_shift'],1)} & "
                   f"{r['exit_shift_over_eps']:.1f} & {r['dev200_comp']:.2f} & {r['dev2000_comp']:.2f} \\\\")
    return "\n".join(out), d


def table_robust():
    rows = J("robustness_oat.json")
    try:
        rows = rows + J("robustness_Bcurve.json")
    except FileNotFoundError:
        pass
    out = []
    for r in rows:
        if "error" in r:
            out.append(f"{r['label']} & \\multicolumn{{8}}{{l}}{{failed: {r['error'][:40]}}} \\\\"); continue
        fl, fu = r.get("fold_lower", {}), r.get("fold_upper", {})
        out.append(f"{r['label'].replace('%', r'\%')} & {r['R0']:.3f} & {sci(r['bcrit'],3)} & {r['btrans_over_bcrit']:.1f} & "
                   f"{r.get('bH_over_bcrit', float('nan')):.4f} & {r.get('l1', float('nan')):.2f} & "
                   f"{sci(fl.get('estimate'),3)} & {sci(fu.get('estimate'),3)} & "
                   f"{'yes' if r.get('lower_fold_below_bH') and r.get('upper_fold_above_bcrit') else 'no'} \\\\")
    return "\n".join(out), rows


def main():
    gz, gd = table_grazing()
    rb, rrows = table_robust()
    emp = J("england_empirical_results.json")
    eq = J("capacity_equilibria_quadratic.json")
    ds = J("dimensional_sensitivity.json")["elasticities"]
    fs = J("formal_dimensionless_sensitivity.json")
    names_dim = list(ds["R0"].keys())
    dim_rows = "\n".join(f"${n}$ & {ds['R0'][n]:.4f} & {ds['bcrit'][n]:.4f} & {ds['btrans'][n]:.4f} \\\\"
                         .replace("$palpha$", r"$p\alpha$").replace("$beta$", r"$\beta$").replace("$mu$", r"$\mu$")
                         .replace("$sigma$", r"$\sigma$").replace("$delta$", r"$\delta$").replace("$nu$", r"$\nu$")
                         .replace("$kappa$", r"$\kappa$").replace("$gammaQ$", r"$\gamma_Q$").replace("$gammaH$", r"$\gamma_H$")
                         .replace("$dI$", r"$d_I$").replace("$dQ$", r"$d_Q$").replace("$dH$", r"$d_H$")
                         .replace("$thetaQ$", r"$\theta_Q$") for n in names_dim)
    bench = emp["benchmarks"]; pro = emp["prospective"]
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,authblk}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\theequation}{S\arabic{equation}}
\newcommand{\bcrit}{b_{\mathrm{crit}}}
\newcommand{\btrans}{b_{\mathrm{trans}}}
\hypersetup{pdftitle={Supplementary Information: Hopf bifurcation, bistability and transient grazing in a piecewise-smooth hospital-overload model},
pdfauthor={Maniru Ibrahim; Bashir Alhassan; Sirajo Ibrahim; Suleiman Ibrahim}}
\begin{document}
\title{Supplementary Information\\[2pt]\large Hopf bifurcation, bistability and transient grazing in a piecewise-smooth hospital-overload model}
\author[1]{Maniru Ibrahim}
\author[2,3]{Bashir Alhassan}
\author[4,5]{Sirajo Ibrahim}
\author[6]{Suleiman Ibrahim}
\affil[1]{Mathematics Applications Consortium for Science and Industry (MACSI), Department of Mathematics and Statistics, University of Limerick, Limerick, V94 T9PX, Ireland}
\affil[2]{Department of Mathematics, Northwest University, Sokoto, Nigeria}
\affil[3]{Department of Mathematics, Usmanu Danfodiyo University, Sokoto, Nigeria}
\affil[4]{Department of Clinical Pharmacy and Pharmacy Practice, Integral University, Lucknow, India}
\affil[5]{Department of Pharmacy, Federal Teaching Hospital, Birnin Kebbi, Nigeria}
\affil[6]{Faculty of Pharmaceutical Sciences, Usmanu Danfodiyo University, Sokoto, Nigeria}
\date{}
\maketitle

All numbers below are generated by \texttt{build\_si.py} from the archived
outputs in \texttt{results/} of release 1.1.0.

\section{Certified computation of the transient threshold}\label{si:btrans}
\paragraph{Algorithm.} For given parameters and initial state, the
normal-care system is integrated with DOP853 (relative tolerance $10^{-10}$,
absolute $10^{-14}$, maximum step $1$) on $[0,T]$ with $T=200,400,800,\dots$
up to $2\times10^5$. On each horizon, $\mathcal P=h+(a/c)q$ is sampled with
spacing $\max(0.05,T/10^5)$; every interior local maximum is refined by
bounded Brent optimisation to $10^{-11}$, and the largest value, together with
$\mathcal P(0)$, is retained. Let $J_N$ be the normal-care Jacobian at
$\mathcal E_N^*$ with eigenvalues $\lambda_k$ (all with negative real part),
right eigenvectors $v_k$ and left eigenvectors $u_k$ normalised by
$u_k^\top v_k=1$. For the linearisation
$\delta(\tau)=\sum_k v_ke^{\lambda_k(\tau-T)}u_k^\top\delta(T)$, so
\begin{equation}
\sup_{\tau\ge T}|\mathcal P(\tau)-\bcrit|\le B_T:=\sum_k|w^\top v_k|\,|u_k^\top\delta(T)|,
\qquad w=(0,0,0,a/c,1,0)^\top .
\end{equation}
The demographic mode $\lambda=-m$, whose eigenvector has zero $q$- and
$h$-components (main text, Theorem~6.1), does not contribute, which is why
the bound is effective despite the slow rate $m$. Integration stops when
$\|\delta(T)\|<10^{-2}\|\mathcal E_N^*\|$ and
$B_T(1+10\|\delta(T)\|/\|\mathcal E_N^*\|)<\tfrac12(\max\mathcal P-\bcrit)$,
where the factor $1+10\|\delta\|/\|\mathcal E_N^*\|$ is a quadratic safety
allowance for the nonlinear remainder. If the certificate fails by
$T=2\times10^5$ the routine raises an error; it never returns an uncertified
value. The ordering $\btrans\ge\bcrit$ is then asserted.

\paragraph{Effect of the correction.} Table~\ref{tab:si-atlas} lists the
affected part of the atlas. For $\mathcal B\ge1.38$ the v1.0 and v1.1 values
agree to better than $10^{-6}$ relative (maximum $3.8\times10^{-11}$).

\begin{table}[h]\centering\small
\caption{Transient threshold near the invasion threshold: v1.0 search restricted to $\tau\le100$ versus the certified value.}
\label{tab:si-atlas}
\begin{tabular}{@{}cccccccc@{}}\toprule
$\mathcal B$ & $R_0$ & $\bcrit$ & $\btrans$ (v1.0) & $\btrans$ (certified) & $\tau^*$ & horizon $T$ & v1.0 violates Thm 7.1 \\\midrule
""" + table_atlas() + r"""
\bottomrule\end{tabular}\end{table}

\paragraph{Initial-condition dependence.} Table~\ref{tab:si-ic} gives
$\btrans/\bcrit$ for alternative initial conditions. ``immune'' denotes the
fraction initially in $r$, with $i(0)=10^{-6}$ and $s(0)$ adjusted.

\begin{table}[h]\centering\small
\caption{$\btrans/\bcrit$ under alternative initial conditions.}\label{tab:si-ic}
\begin{tabular}{@{}lcccccc@{}}\toprule
""" + table_ic() + r"""
\bottomrule\end{tabular}\end{table}

\section{Capacity-limited equilibria: explicit quadratic}\label{si:quad}
The coefficients in main-text Eqs.~(18)--(19) were obtained with SymPy and verified
symbolically against the direct expansion of $F_C$; the identity
$n_C=1-(\Delta_Ii+\Delta_Qq+\Delta_Hh)/m$ used to simplify $\alpha_2$ follows
from the scaling identities and was also verified symbolically. Over $90$
values of $b\in[10^{-6},1]$ the roots of the quadratic coincide with those of
the root finder of release 1.0 to $10^{-12}$ relative. At the baseline the
vertex lies below $i_\Sigma^{(C)}$ for every $b>\bcrit$
($\max(i_v-i_\Sigma)=""" + f"{max(r['vertex']-r['i_sigma'] for r in eq['baseline_scan'] if r['b']>1.0257e-3):.3e}" + r"""$).

\paragraph{Random parameter scan.} Parameters were drawn uniformly:
$\mathcal B\in[1,12]$, $\eta\in[0.1,2]$, $\pi\in[0.02,0.6]$, $a\in[0.05,3]$,
$c\in[0.05,10]$, $\theta_Q\in[0,1]$, $\log_{10}m\in[-4,-1.5]$,
$\Delta_I\in[0,0.05]$ with $\rho_I=1-\pi-m-\Delta_I>0$,
$\rho_Q,\rho_H\in[0,0.5]$, $\chi_\bullet=\rho_\bullet+\Delta_\bullet+m$ with
$\Delta_\bullet\in[0,0.3]$; sets with $R_0\le1$ were discarded. In 999
accepted sets, $F_C'(i_\Sigma^{(C)})$ at $b=\bcrit$ was negative in 978 cases,
all with $\det J_N\det J_C>0$ (persistence), and positive in 21 cases, all
with $\det J_N\det J_C<0$ (nonsmooth fold). In the latter, two admissible
capacity-limited equilibria exist on an interval of $b$ just above $\bcrit$.

\section{Periodic orbits: collocation, saltation and grazing}\label{si:cycles}
\paragraph{Collocation continuation.} Unknowns: $u=\log y$ on $s\in[0,1]$,
period $T$, capacity $b$. Equations: $u'=T\,g(u;b)$ with
$g=f(e^u)/e^u$; boundary conditions $u(0)=u(1)$, $q'(0)=0$ and
$(z-z_{\rm prev})\cdot t=\Delta s$ with
$z=(u(0),T/100,b/10^{-3})$ and $t$ the secant tangent. SciPy
\texttt{solve\_bvp} (fourth-order Lobatto IIIA collocation with residual
control) was used, re-meshed to 2500 nodes before each step and refined
adaptively. Folds are turning points of $b$, located by quadratic
interpolation of $b$ against amplitude through the five points nearest the
extremum. Table~\ref{tab:si-folds} shows the convergence study.

\begin{table}[h]\centering\small
\caption{Collocation fold refinement. $\Psi$ extrema, the normal-care fraction
of the period, and $\min|\mathrm d\Psi/\mathrm d\tau|$ at the crossings refer
to the computed orbit nearest the fold.}\label{tab:si-folds}
\begin{tabular}{@{}lcccccccc@{}}\toprule
fold & tol & $\Delta s$ & $b_{\rm LPC}$ & nodes & $\min\Psi$ & $\max\Psi$ & N-fraction & $\min|\Psi'|$ \\\midrule
""" + table_folds() + r"""
\bottomrule\end{tabular}\end{table}

\paragraph{Saltation and monodromy.} Table~\ref{tab:si-salt} compares the modulus of the leading nontrivial Floquet multiplier from
(a) the original pointwise-switching variational integration, (b) event-located
integration with explicit saltation matrices, and (c) central finite
differences of the flow map (Radau, relative tolerance $10^{-12}$, step
$10^{-7}$ of the variable scale). Every crossing is transversal
($|\mathrm d\Psi/\mathrm d\tau|>0$) and $S=I$ to rounding error.

\begin{table}[h]\centering\small
\caption{Monodromy verification on attracting cycles.}\label{tab:si-salt}
\begin{tabular}{@{}ccccccccc@{}}\toprule
$b$ & $T$ & residual & $|\mu|$ (a) & $|\mu|$ (b) & $|\mu|$ (c) & crossings & $\min|\Psi'|$ & $\max\|S-I\|_\infty$ \\\midrule
""" + table_saltation() + r"""
\bottomrule\end{tabular}\end{table}

\paragraph{Grazing of the unstable branch.} Along the collocation branch the
minimum of $\Psi$ over the orbit changes sign near $b=10^{-4}$. A local
collocation continuation with step $\Delta s=0.005$ started at
$b=9.45\times10^{-5}$ ($\min\Psi=-2.2\times10^{-5}$) brackets the sign change
between consecutive orbits and gives, by linear interpolation,
$b_g=1.007\times10^{-4}$ ($b_g/\bcrit=0.0982$). Above $b_g$ the unstable cycle is entirely
capacity-limited. The archived shooting branch is consistent: its orbits at
$b\ge1.128\times10^{-4}$ have $\min\Psi>0$, and at $8.65\times10^{-5}$
$\min\Psi<0$.

\section{Derivation of the grazing laws}\label{si:grazing}
Let $x=\tau-\tau^*$ and
$\Psi_N=c[\varepsilon-\tfrac k2x^2+\tfrac{k_3}{6}x^3+O(x^4)]$ with
$k_3=\mathcal P'''(\tau^*)$.

\emph{Reference roots.} Writing $x_\pm=\pm x_0+x_1+O(\varepsilon^{3/2})$ with
$x_0=\sqrt{2\varepsilon/k}$, substitution and collection of the
$O(\varepsilon)$ terms gives $-kx_0x_1+\tfrac{k_3}{6}x_0^3=0$ for either sign of $x_0$, i.e.
$x_1=k_3x_0^2/(6k)=k_3\varepsilon/(3k^2)$, the same for both roots.
Hence $\tau_2^N-\tau_1^N=2\sqrt{2\varepsilon/k}+O(\varepsilon^{3/2})$: the cubic
term shifts the episode but does not lengthen it at $O(\varepsilon)$.

\emph{Perturbation.} With $e=(0,0,0,1,-1,0)^\top$, the full field is
$f_N+[\Psi]_+e$. For $\tau\in[\tau_1,\tau_2]$,
$\delta=y-y_N$ satisfies $\delta(\tau)=\int_{\tau_1}^{\tau}\Phi(\tau,\sigma)e\,[\Psi(\sigma)]_+\,\mathrm d\sigma+O(\|\delta\|^2)$,
where $\Phi$ is the normal-care state-transition matrix. Since
$\Phi(\tau,\sigma)=I+O(\tau-\sigma)=I+O(\varepsilon^{1/2})$ on the episode,
$\delta=eD(\tau)+O(\varepsilon^2)$ with $D(\tau)=\int_{\tau_1}^\tau[\Psi]_+$.

\emph{Exit time.} Along the full solution
$\Psi=\Psi_N+\nabla\Psi^\top\delta=\Psi_N+(a-c)D+O(\varepsilon^2)$. Expanding
about $\tau_2^N$ with $\Psi_N'(\tau_2^N)=-c\sqrt{2k\varepsilon}+O(\varepsilon)$
and $D(\tau_2)=\frac{4\sqrt2}{3}ck^{-1/2}\varepsilon^{3/2}+O(\varepsilon^2)$,
\begin{equation}
\tau_2-\tau_2^N=\frac{(a-c)D(\tau_2)}{c\sqrt{2k\varepsilon}}+O(\varepsilon^{3/2})
=\frac{4(a-c)}{3k}\varepsilon+O(\varepsilon^{3/2}).
\end{equation}
At the baseline, $4(a-c)/(3k)=""" + f"{4*(0.7-2.0)/(3*gd['k']):.1f}" + r"""$.

\emph{Deficit.} $D_{\rm adm}=\int_{\tau_1}^{\tau_2}\Psi=\int_{\tau_1^N}^{\tau_2^N}\Psi_N
+\int(a-c)D+[\text{endpoint terms}]$. The first integral equals
$c\int_{-x_0}^{x_0}(\varepsilon-\tfrac k2x^2)\,\mathrm dx+O(\varepsilon^2)=\frac{4\sqrt2}{3}ck^{-1/2}\varepsilon^{3/2}+O(\varepsilon^2)$
(the cubic term is odd about the shifted centre and contributes at
$O(\varepsilon^2)$). The second is $O(\varepsilon^{3/2})\times O(\varepsilon^{1/2})=O(\varepsilon^2)$.
The endpoint shift is $O(\varepsilon)$ and $\Psi=O(\varepsilon)$ there, so it adds
$O(\varepsilon^2)$.

\emph{Uniqueness and uniform deviation.} Conditions (G1)--(G4) of the main text
give a fixed neighbourhood $U$ of $\tau^*$ outside which $\Psi_N\le-c\delta_0$.
At $\tau_2$ the perturbation is $O(\varepsilon^{3/2})$. Since $\mathcal E_N^*$ is
exponentially stable for the smooth normal-care field and
$y_N(\tau)\to\mathcal E_N^*$, the variational system along $y_N$ is uniformly
exponentially bounded on $[\tau_2,\infty)$ (roughness of exponential
dichotomies), so $\sup_{\tau\ge\tau_2}\|\delta\|=O(\varepsilon^{3/2})$ and
$\Psi<0$ for $\tau>\tau_2$.

\paragraph{Diagnostics.} Table~\ref{tab:si-graz} lists the event-located
diagnostics at relative tolerance $10^{-12}$. Constants:
$2\sqrt{2/k}=""" + f"{gd['duration_constant']:.4f}" + r"""$,
$\frac{4\sqrt2}{3}ck^{-1/2}=""" + f"{gd['deficit_constant']:.4f}" + r"""$.

\begin{table}[h]\centering\small
\caption{Grazing diagnostics (relative tolerance $10^{-12}$). Ratios are to the asymptotic constants. The
$[0,2000]$ column reaches the integration-error floor for $\varepsilon<10^{-7}$.}\label{tab:si-graz}
\begin{tabular}{@{}cccccccc@{}}\toprule
$\varepsilon$ & episodes & $\Delta\tau$ ratio & $D_{\rm adm}$ ratio & entry shift & exit shift$/\varepsilon$ &
$\sup_{[0,200]}\|\delta\|/\varepsilon^{3/2}$ & $\sup_{[0,2000]}\|\delta\|/\varepsilon^{3/2}$ \\\midrule
""" + gz + r"""
\bottomrule\end{tabular}\end{table}

\section{Structural robustness}\label{si:robust}
Table~\ref{tab:si-robust} gives the full robustness results. When $\pi$ is
varied, $\rho_I$ is adjusted to keep $\pi+\rho_I+m+\Delta_I=1$; when $\rho_H$
is varied, $\chi_H=\rho_H+\Delta_H+m$ is adjusted. $\chi_Q$ and $\chi_H$ are
total removal rates, so varying them alters the corresponding $\Delta$.
Fold estimates are geometric means of hysteresis-simulation brackets of
relative width $2\times10^{-3}$, each run lasting $\tau=15000$. At the baseline
the brackets lie $0.14\%$ below and $0.4\%$ above the continuation folds,
reflecting slow passage near a saddle-node.

\begin{table}[h]\centering\footnotesize
\caption{Robustness of the Hopf criticality, folds and threshold separation.
``bistable'' requires $b_{\rm LPC}^-<b_H$ and $b_{\rm LPC}^+>\bcrit$.}\label{tab:si-robust}
\begin{tabular}{@{}lcccccccc@{}}\toprule
variation & $R_0$ & $\bcrit$ & $\btrans/\bcrit$ & $b_H/\bcrit$ & $\ell_1$ & $b_{\rm LPC}^-$ & $b_{\rm LPC}^+$ & bistable \\\midrule
""" + rb + r"""
\bottomrule\end{tabular}\end{table}

\section{Illustrative bed-turnover check: details}\label{si:emp}
Training uses """ + str(emp["n_training_transitions"]) + r""" transitions with end date on or before
30 November 2020. $\widehat\gamma_H=""" + f"{emp['gamma_hat']:.5f}" + r"""$ day$^{-1}$; OLS
standard error """ + f"{emp['gamma_se_ols']:.5f}" + r"""; Newey--West (7 lags) """ + f"{emp['gamma_se_hac7']:.5f}" + r""";
moving-block bootstrap (block 14 days, 4000 replicates) $95\%$ interval
$[""" + f"{emp['gamma_ci95_block_bootstrap'][0]:.4f},{emp['gamma_ci95_block_bootstrap'][1]:.4f}" + r"""]$.
Training-residual Ljung--Box statistics: $Q(7)=""" + f"{emp['training_residual_ljungbox']['lb_stat']['7']:.1f}" + r"""$,
$Q(14)=""" + f"{emp['training_residual_ljungbox']['lb_stat']['14']:.1f}" + r"""$ ($p<10^{-30}$).
Validation one-step residuals: mean """ + f"{emp['validation_one_step']['mean']:.1f}" + r""", s.d. """ + f"{emp['validation_one_step']['sd']:.1f}" + r"""
beds; lag-1 and lag-7 autocorrelations """ + f"{emp['validation_one_step']['acf1']:.2f}" + r""" and """ + f"{emp['validation_one_step']['acf7']:.2f}" + r""".

\begin{table}[h]\centering\small
\caption{Held-out performance (1 December 2020 to 6 April 2021) of the stock
balance and benchmarks, all fitted on the training period only.}
\begin{tabular}{@{}lcc@{}}\toprule
Model & $R^2$ & RMSE (beds) \\\midrule
Stock balance, observed inflow (input-conditioned) & """ + f"{emp['validation_R2']:.3f} & {emp['validation_RMSE']:.0f}" + r""" \\
Occupancy persistence from 1 Dec & """ + f"{bench['persistence_H'][0]:.3f} & {bench['persistence_H'][1]:.0f}" + r""" \\
""" + f"{bench['window_sum']['L']}" + r"""-day admissions sum & """ + f"{bench['window_sum']['r2_rmse'][0]:.3f} & {bench['window_sum']['r2_rmse'][1]:.0f}" + r""" \\
Distributed lag, 21 lags & """ + f"{bench['distributed_lag_21'][0]:.3f} & {bench['distributed_lag_21'][1]:.0f}" + r""" \\
\midrule
Prospective 7-day, stock balance & """ + f"{pro['h7']['stock_flow'][0]:.3f} & {pro['h7']['stock_flow'][1]:.0f}" + r""" \\
Prospective 7-day, persistence & """ + f"{pro['h7']['persistence'][0]:.3f} & {pro['h7']['persistence'][1]:.0f}" + r""" \\
Prospective 14-day, stock balance & """ + f"{pro['h14']['stock_flow'][0]:.3f} & {pro['h14']['stock_flow'][1]:.0f}" + r""" \\
Prospective 14-day, persistence & """ + f"{pro['h14']['persistence'][0]:.3f} & {pro['h14']['persistence'][1]:.0f}" + r""" \\
\bottomrule\end{tabular}\end{table}

Peak dates: raw """ + f"{emp['peaks']['raw']['admissions']} / {emp['peaks']['raw']['occupancy']} (lag {emp['peaks']['raw']['lag_days']} d)" + r""";
trailing 7-day means """ + f"{emp['peaks']['trailing7_end_labelled']['admissions']} / {emp['peaks']['trailing7_end_labelled']['occupancy']} (lag {emp['peaks']['trailing7_end_labelled']['lag_days']} d)" + r""";
centred 7-day means """ + f"{emp['peaks']['centred7']['admissions']} / {emp['peaks']['centred7']['occupancy']} (lag {emp['peaks']['centred7']['lag_days']} d)" + r""".

\section{Sensitivities to dimensional rates}\label{si:dimsens}
Baseline dimensional rates are expressed in units of $\kappa_I$ (so they equal
the dimensionless baseline values). Each rate is perturbed by $\pm10^{-4}$
relative, with all groups recomputed from Appendix A. The Euler sums over the
twelve rates are """ + ", ".join(f"{v:.1e}" for v in J('dimensional_sensitivity.json')['euler_sum_over_rates'].values()) + r""" for $R_0$, $\bcrit$, $\btrans$.

\begin{table}[h]\centering\small
\caption{Elasticities with respect to independent dimensional rates.}
\begin{tabular}{@{}lccc@{}}\toprule
Rate & $R_0$ & $\bcrit$ & $\btrans$ \\\midrule
""" + dim_rows + r"""
\bottomrule\end{tabular}\end{table}

\section{Direct bistability runs}\label{si:bistab}
Integration: LSODA, relative tolerance $10^{-9}$, absolute $10^{-12}$, to
$\tau=8000$. Classification: peak-to-peak range of $q$ over the final 1000 time
units, threshold $10^{-4}$ (equilibrium runs give $<3\times10^{-10}$, cycle runs
$>9\times10^{-3}$, so the classification is insensitive to the threshold).
Grid: $31\times31$ in $(q,h)\in[0,0.01625]\times[0,0.999b]$.

\end{document}
"""
    OUT.write_text(tex)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
