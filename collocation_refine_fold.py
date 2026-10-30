#!/usr/bin/env python3
"""Local refinement of a fold of cycles by collocation pseudo-arclength
continuation with small steps, started from an orbit saved by
collocation_folds.py.  Repeated for several collocation tolerances and step
sizes to demonstrate convergence of the fold location (R2.2).

Usage: python collocation_refine_fold.py lower|upper
Output: results/collocation_fold_refine_{lower|upper}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import collocation_folds as cf
from verify_revision import rhs_full

OUT = Path(__file__).resolve().parent / "results"


def profile_from_row(row, npts=2500):
    b, T = row["b"], row["T"]
    sol = solve_ivp(lambda t, y: rhs_full(t, y, b), (0, T), np.array(row["y0"]), method="DOP853",
                    rtol=1e-12, atol=1e-16, dense_output=True)
    s = np.linspace(0, 1, npts)
    U = np.log(sol.sol(s * T))
    U[:, -1] = U[:, 0]
    return s, U


def refine(row, tol, ds, nsteps, direction):
    br = cf.Branch(tol=tol)
    s, U = profile_from_row(row)
    sol, T, b = br.solve(s, U, row["T"], row["b"], fixed_b=True)
    assert sol.success, sol.message
    z1 = np.r_[sol.sol(0.0), T / cf.TS, b / cf.BS]
    tv = np.zeros_like(z1); tv[-1] = direction
    sol2, T2, b2 = br.solve(sol.x, sol.y, T, b + direction * ds * cf.BS, z_prev=z1, tangent=tv, ds=ds)
    assert sol2.success, sol2.message
    pts = [(sol, T, b), (sol2, T2, b2)]
    rows = [cf.orbit_summary(*pts[0]), cf.orbit_summary(*pts[1])]
    for k in range(nsteps):
        (s1, T1, b1), (s2, T2, b2) = pts[-2], pts[-1]
        za = np.r_[s1.sol(0.0), T1 / cf.TS, b1 / cf.BS]
        zb = np.r_[s2.sol(0.0), T2 / cf.TS, b2 / cf.BS]
        tv = (zb - za) / np.linalg.norm(zb - za)
        xg = np.linspace(0, 1, 2500)
        fac = ds / np.linalg.norm(zb - za)
        Up = s2.sol(xg) + fac * (s2.sol(xg) - s1.sol(xg))
        sol, T, b = br.solve(xg, Up, T2 + fac * (T2 - T1), b2 + fac * (b2 - b1), z_prev=zb, tangent=tv, ds=ds)
        if not sol.success:
            print("  step failed", sol.message); break
        pts.append((sol, T, b)); pts = pts[-2:]
        rows.append(cf.orbit_summary(sol, T, b))
    bs = np.array([r["b"] for r in rows]); am = np.array([r["ampq"] for r in rows])
    j = int(np.argmax(bs)) if direction > 0 else int(np.argmin(bs))
    if 0 < j < len(rows) - 1:
        lo, hi = max(0, j - 2), min(len(rows), j + 3)
        cfit = np.polyfit(am[lo:hi], bs[lo:hi], 2)
        xv = -cfit[1] / (2 * cfit[0]); bf = float(np.polyval(cfit, xv))
    else:
        xv, bf = float("nan"), float("nan")
    return dict(tol=tol, ds=ds, b_fold=bf, ampq_fold=float(xv), b_extreme_sample=float(bs[j]),
                interior=bool(0 < j < len(rows) - 1), fold_row=rows[j],
                nodes_at_fold=rows[j]["nodes"], rows=rows)


def main(which):
    tag = "m1" if which == "lower" else "p1"
    d = json.load(open(OUT / f"collocation_branch_tol1e-06_dir{-1 if which == 'lower' else 1}.json")) \
        if (OUT / f"collocation_branch_tol1e-06_dir{-1 if which == 'lower' else 1}.json").exists() else None
    rows = d["rows"] if d else json.load(open(OUT / f"partial_{tag}.json"))
    bs = np.array([r["b"] for r in rows])
    if which == "upper":
        j = int(np.argmax(bs)); start = rows[max(0, j - 2)]; direction = +1
    else:
        j = int(np.argmin(bs)); start = rows[max(0, j - 2)]; direction = -1
    out = []
    for tol, ds, n in [(1e-5, 0.01, 40), (1e-6, 0.01, 40), (1e-6, 0.005, 70), (3e-7, 0.01, 40)]:
        r = refine(start, tol, ds, n, direction)
        print(which, f"tol={tol:g} ds={ds:g} b_fold={r['b_fold']:.8e} interior={r['interior']} "
              f"nodes={r['nodes_at_fold']} psi=[{r['fold_row']['psimin']:.3e},{r['fold_row']['psimax']:.3e}] "
              f"fracC={r['fold_row']['fracC']:.3f} min|dPsi|={r['fold_row']['min_abs_dpsi_at_cross']}", flush=True)
        r.pop("rows")
        out.append(r)
        json.dump(out, open(OUT / f"collocation_fold_refine_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
