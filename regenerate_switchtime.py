#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from verify_revision import P0, I0, normal_equilibrium, transient_threshold, first_root

OUT = Path(__file__).resolve().parent
_, bcrit = normal_equilibrium()
btrans, _, sol = transient_threshold(P0, 160.0)
a, c = P0[4], P0[5]

def pressure(t):
    y = sol.sol(t)
    return float(y[4] + a / c * y[3])

B, m, eta, pi, a, c, chiQ, chiH, rhoI, rhoQ, rhoH, theta = map(float, P0)
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
ax.legend(fontsize=8, loc="upper left")
ax2 = ax.twinx()
ax2.semilogx(bvals, 100.0 * relerr, linestyle=":", label="relative gap")
ax2.set_ylabel("bound gap (%)")
fig.tight_layout()
fig.savefig(OUT / "fig_switchtime.pdf", bbox_inches="tight")
plt.close(fig)
print(OUT / "fig_switchtime.pdf")
print(f"bcrit={bcrit:.12g}, btrans={btrans:.12g}, max_gap={100*np.nanmax(relerr):.3f}%")
