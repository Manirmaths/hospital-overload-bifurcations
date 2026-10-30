#!/usr/bin/env python3
"""R2.7: capacity-limited equilibria as roots of an explicit quadratic.

Using n = 1 - (Delta_I i + Delta_Q q + Delta_H h)/m at any equilibrium, the
scalar equation F_C(i)=0 (manuscript Eq. 17) reduces to
    F_C(i) = alpha2 i^2 + alpha1 i + alpha0,
with coefficients verified symbolically below and checked numerically against
the root finder used in v1.0.0.
"""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from verify_revision import P0, normal_equilibrium, capacity_equilibria, R0_value

OUT = Path(__file__).resolve().parent / "results"

Bt, m, eta, pi, a, c, chQ, chH, rI, rQ, rH, th, b, i = sp.symbols(
    'B m eta pi a c chiQ chiH rhoI rhoQ rhoH theta b i', positive=True)
L = (eta + m) / (m * eta)
hC = c * b / (c + chH)
AC = c * chH * b / (c + chH)
q = (pi * i - AC) / chQ
s = 1 - L * i
r = (rI * i + rQ * q + rH * hC) / m
n = s + i / eta + i + q + hC + r
F = Bt * s * (i + th * q) - (eta + m) / eta * i * n
# Delta's from the scaling identities
DI = 1 - pi - rI - m
DQ = chQ - rQ - m
DH = chH - rH - m
n_alt = 1 - (DI * i + DQ * q + DH * hC) / m
assert sp.simplify(sp.expand(n - n_alt)) == 0
Fp = sp.Poly(sp.expand(F), i)
al2, al1, al0 = [sp.simplify(x) for x in Fp.all_coeffs()]
# closed forms written in the manuscript
KQ = 1 + th * pi / chQ
al2_ms = -L * (Bt * KQ - DI - DQ * pi / chQ)
al0_ms = -Bt * th * AC / chQ
al1_ms = (Bt * KQ - (eta + m) / eta * (1 + DQ * AC / (m * chQ) - DH * hC / m)) \
    + L * Bt * th * AC / chQ
for x, y in [(al2, al2_ms), (al0, al0_ms), (al1, al1_ms)]:
    assert sp.simplify(sp.expand(x - y)) == 0, (x, y)
print("alpha2 =", al2_ms); print("alpha1 =", al1_ms); print("alpha0 =", al0_ms)

# F_C at the switching value i_Sigma = A_C (a+chiQ)/(a pi) is kappa1 b (1 - b/bcrit)
iS = AC * (a + chQ) / (a * pi)
FS = sp.factor(sp.simplify(F.subs(i, iS)))
print("F_C(i_Sigma) =", FS)

vals = dict(zip([Bt, m, eta, pi, a, c, chQ, chH, rI, rQ, rH, th], map(float, P0)))
yN, bc = normal_equilibrium()
fS = sp.lambdify(b, FS.subs(vals))
print("check root at bcrit:", fS(bc), " sign below:", np.sign(fS(0.5 * bc)), " above:", np.sign(fS(2 * bc)))
k1 = sp.limit(FS / b, b, 0)
print("kappa1 =", sp.simplify(k1))

# numerical cross-check vs v1.0.0 root finder, and scan b > bcrit for spurious roots
fa = [sp.lambdify(b, x.subs(vals)) for x in (al2, al1, al0)]
iSf = sp.lambdify(b, iS.subs(vals)); Lf = float(L.subs(vals))
rows = []
for bb in np.r_[np.geomspace(1e-6, bc * 0.999, 30), np.geomspace(bc * 1.001, 1.0, 60)]:
    A2, A1, A0 = fa[0](bb), fa[1](bb), fa[2](bb)
    disc = A1 ** 2 - 4 * A2 * A0
    roots = np.roots([A2, A1, A0])
    adm = [float(x.real) for x in roots if abs(x.imag) < 1e-14 and iSf(bb) <= x.real < 1 / Lf]
    old = capacity_equilibria(bb)
    rows.append(dict(b=float(bb), alpha2=A2, alpha1=A1, alpha0=A0, disc=disc,
                     vertex=-A1 / (2 * A2), i_sigma=iSf(bb), n_adm=len(adm), n_old=len(old)))
    if old:
        assert abs(old[0][2] - adm[0]) < 1e-12 * max(1, adm[0]), (bb, old[0][2], adm)
print("baseline: number of admissible roots (b<bcrit):", {r['n_adm'] for r in rows if r['b'] < bc},
      " (b>bcrit):", {r['n_adm'] for r in rows if r['b'] > bc})

# Parameter scan: can two admissible capacity-limited equilibria coexist for b > bcrit?
rng = np.random.default_rng(1)
names = ["B", "m", "eta", "pi", "a", "c", "chiQ", "chiH", "rhoI", "rhoQ", "rhoH", "thetaQ"]
two = 0; tested = 0; examples = []
for trial in range(4000):
    p = P0.copy()
    p[0] = rng.uniform(1.0, 12.0); p[2] = rng.uniform(0.1, 2.0); p[3] = rng.uniform(0.02, 0.6)
    p[4] = rng.uniform(0.05, 3.0); p[5] = rng.uniform(0.05, 10.0); p[11] = rng.uniform(0, 1.0)
    p[1] = 10 ** rng.uniform(-4, -1.5)
    dI = rng.uniform(0, 0.05); p[8] = 1 - p[3] - p[1] - dI
    if p[8] <= 0: continue
    p[9] = rng.uniform(0.0, 0.5); p[6] = p[9] + rng.uniform(0, 0.3) + p[1]
    p[10] = rng.uniform(0.0, 0.5); p[7] = p[10] + rng.uniform(0, 0.3) + p[1]
    if R0_value(p) <= 1: continue
    v = dict(zip([Bt, m, eta, pi, a, c, chQ, chH, rI, rQ, rH, th], map(float, p)))
    f2 = [sp.lambdify(b, x.subs(v)) for x in (al2, al1, al0)]
    _, bcp = normal_equilibrium(p)
    iS2 = sp.lambdify(b, iS.subs(v)); L2 = float(L.subs(v))
    tested += 1
    for bb in bcp * np.geomspace(1.0001, 1e3, 40):
        if bb * float((c / (c + chH)).subs(v)) > 1: break
        rts = np.roots([f2[0](bb), f2[1](bb), f2[2](bb)])
        adm = [x.real for x in rts if abs(x.imag) < 1e-14 and iS2(bb) <= x.real < 1 / L2]
        if len(adm) >= 1:
            two += 1; examples.append(dict(p=p.tolist(), b=float(bb), bcrit=float(bcp), roots=adm)); break
    if trial % 200 == 0: pass
    if tested >= 300: break
print(f"random scan: {tested} parameter sets, {two} with admissible capacity-limited roots for b>bcrit")
json.dump(dict(alpha2=str(al2_ms), alpha1=str(al1_ms), alpha0=str(al0_ms), F_at_iSigma=str(FS),
               kappa1=str(sp.simplify(k1)), baseline_scan=rows, random_tested=tested,
               random_with_roots_above_bcrit=two, examples=examples[:5]),
          open(OUT / "capacity_equilibria_quadratic.json", "w"), indent=1, default=float)
