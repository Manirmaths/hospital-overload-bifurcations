#!/usr/bin/env python3
"""NHS England hospital-stock illustration (revision 1.1.0).

Changes from v1.0.0 (Reviewer 2, comments 9-10; Reviewer 1, comment 2)
--------------------------------------------------------------------------
* strict train/validation separation: a training transition t -> t+1 is used
  only if BOTH H_t and H_{t+1} are dated on or before 30 Nov 2020; the
  validation recursion is initialised at the observed 1 Dec value, which is no
  longer used in fitting;
* uncertainty for gamma_H: OLS, Newey-West (HAC, 7 lags) and moving-block
  bootstrap (block 14 d, 4000 replicates) intervals;
* residual diagnostics: one-step residual mean, lag-1/lag-7 autocorrelation,
  Ljung-Box test;
* benchmarks on the same held-out window: (i) occupancy persistence,
  (ii) fixed-window admissions sum H_t = beta sum_{j<L} A_{t-j},
  (iii) distributed-lag regression on A_t ... A_{t-20},
  plus genuinely prospective 7- and 14-day-ahead forecasts in which future
  inflow is held at its last observed value;
* constancy of gamma_H: rolling 28-day estimates;
* peak timing reported separately for raw series, trailing 7-day means
  (labelled by window end, as plotted) and centred 7-day means.

Interpretation: input-conditioned reconstruction of the hospital stock from
an observed inflow proxy. It does not test the admission switch Psi, the
latent queue Q, or the thresholds b_crit and b_trans.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CSV_PATH = ROOT / "england_covid_hospital_2020-08-01_to_2021-04-06.csv"
TRAIN_END = datetime(2020, 11, 30)
VAL_START = datetime(2020, 12, 1)


def trailing7(x):
    return np.convolve(x, np.ones(7) / 7, mode="valid")


def r2_rmse(obs, pred):
    res = obs - pred
    return float(1 - np.sum(res ** 2) / np.sum((obs - obs.mean()) ** 2)), float(np.sqrt(np.mean(res ** 2)))


def recursive(H0, A, gamma):
    pred = np.empty(len(A) + 1)
    pred[0] = H0
    for k in range(len(A)):
        pred[k + 1] = pred[k] + A[k] - gamma * pred[k]
    return pred


def main():
    dates, adm, occ = [], [], []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            adm.append(float(row["total_reported_admissions_and_diagnoses"]))
            occ.append(float(row["occupied_covid_beds_0800"]))
    dates = np.array(dates, dtype=object); adm = np.array(adm); occ = np.array(occ)
    A7, H7, d7 = trailing7(adm), trailing7(occ), dates[6:]

    # ---- balance H_{t+1}-H_t = A_t - gamma H_t ---------------------------------
    resp = A7[:-1] - (H7[1:] - H7[:-1])        # = gamma H_t + error
    Ht = H7[:-1]
    t_from, t_to = d7[:-1], d7[1:]
    train = np.array([d <= TRAIN_END for d in t_to])      # strict: H_{t+1} <= 30 Nov
    train_old = np.array([d <= TRAIN_END for d in t_from])  # v1.0.0 rule
    g_old = float(Ht[train_old] @ resp[train_old] / (Ht[train_old] @ Ht[train_old]))
    ols = sm.OLS(resp[train], Ht[train]).fit()
    hac = sm.OLS(resp[train], Ht[train]).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    g = float(ols.params[0])
    # moving-block bootstrap of the training transitions
    rng = np.random.default_rng(20261026)
    y_tr, x_tr = resp[train], Ht[train]
    n, L = len(y_tr), 14
    boots = []
    for _ in range(4000):
        idx = np.concatenate([np.arange(s, s + L) for s in rng.integers(0, n - L + 1, size=n // L + 1)])[:n]
        boots.append(x_tr[idx] @ y_tr[idx] / (x_tr[idx] @ x_tr[idx]))
    boot_ci = np.percentile(boots, [2.5, 97.5])
    res_tr = y_tr - g * x_tr
    lb_tr = acorr_ljungbox(res_tr, lags=[7, 14], return_df=True)

    # ---- held-out recursive reconstruction --------------------------------------
    s = int(np.where(d7 == VAL_START)[0][0])
    obs = H7[s:]
    vdates = d7[s:]
    pred = recursive(H7[s], A7[s:-1], g)
    r2, rmse = r2_rmse(obs, pred)
    mae = float(np.mean(np.abs(obs - pred)))
    # gamma CI propagated to the reconstruction
    pred_lo = recursive(H7[s], A7[s:-1], float(boot_ci[1]))
    pred_hi = recursive(H7[s], A7[s:-1], float(boot_ci[0]))
    # one-step-ahead residuals in validation
    one_step = (H7[s + 1:] - H7[s:-1]) - (A7[s:-1] - g * H7[s:-1])
    lb_val = acorr_ljungbox(one_step, lags=[7, 14], return_df=True)
    acf1 = float(np.corrcoef(one_step[:-1], one_step[1:])[0, 1])
    acf7 = float(np.corrcoef(one_step[:-7], one_step[7:])[0, 1])

    # ---- benchmarks (fitted on training data only) ------------------------------
    bench = {}
    # (i) persistence of occupancy at its 1 Dec value
    bench["persistence_H"] = r2_rmse(obs, np.full_like(obs, H7[s]))
    # (ii) fixed-window admissions sum, window chosen in training
    best = None
    tr_idx = np.where(d7 <= TRAIN_END)[0]
    for Lw in range(3, 29):
        S = np.array([A7[max(0, k - Lw + 1):k + 1].sum() for k in range(len(A7))])
        ok = tr_idx[tr_idx >= Lw]
        beta = float(S[ok] @ H7[ok] / (S[ok] @ S[ok]))
        sse = float(np.sum((H7[ok] - beta * S[ok]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, Lw, beta, S)
    _, Lw, beta, S = best
    bench["window_sum"] = dict(L=Lw, beta=beta, r2_rmse=r2_rmse(obs, beta * S[s:]))
    # (iii) distributed lag regression on A_t..A_{t-20} (ridge-free OLS, training only)
    lags = 21
    X = np.column_stack([np.r_[np.full(j, np.nan), A7[:len(A7) - j]] for j in range(lags)])
    okr = np.array([k for k in tr_idx if k >= lags])
    coef, *_ = np.linalg.lstsq(X[okr], H7[okr], rcond=None)
    bench["distributed_lag_21"] = r2_rmse(obs, X[s:] @ coef)
    # (iv) prospective h-step forecasts with inflow held at last observed value
    prosp = {}
    for h in (7, 14):
        f_s, o_s, pers = [], [], []
        for k in range(s, len(H7) - h):
            Hp = H7[k]
            for _ in range(h):
                Hp = Hp + A7[k] - g * Hp          # future inflow unknown: hold A_k
            f_s.append(Hp); o_s.append(H7[k + h]); pers.append(H7[k])
        f_s, o_s, pers = map(np.array, (f_s, o_s, pers))
        prosp[f"h{h}"] = dict(stock_flow=r2_rmse(o_s, f_s), persistence=r2_rmse(o_s, pers),
                              mae_stock_flow=float(np.mean(np.abs(o_s - f_s))),
                              mae_persistence=float(np.mean(np.abs(o_s - pers))))

    # ---- constancy of gamma: rolling 28-day estimates --------------------------
    roll = []
    for k in range(0, len(Ht) - 28 + 1, 1):
        x, y = Ht[k:k + 28], resp[k:k + 28]
        gk = float(x @ y / (x @ x))
        se = float(np.sqrt(np.sum((y - gk * x) ** 2) / 27 / (x @ x)))
        roll.append((t_to[k + 27].strftime("%Y-%m-%d"), gk, se))
    g_val_period = float(Ht[~train] @ resp[~train] / (Ht[~train] @ Ht[~train]))

    # ---- peak timing ------------------------------------------------------------
    centred_A = np.convolve(adm, np.ones(7) / 7, mode="same")[3:-3]
    centred_H = np.convolve(occ, np.ones(7) / 7, mode="same")[3:-3]
    dc = dates[3:-3]
    def pk(x, d): return d[int(np.argmax(x))]
    peaks = {
        "raw": (pk(adm, dates), pk(occ, dates)),
        "trailing7_end_labelled": (pk(A7, d7), pk(H7, d7)),
        "centred7": (pk(centred_A, dc), pk(centred_H, dc)),
    }
    peak_out = {k: dict(admissions=v[0].strftime("%Y-%m-%d"), occupancy=v[1].strftime("%Y-%m-%d"),
                        lag_days=(v[1] - v[0]).days) for k, v in peaks.items()}

    res = dict(
        source_file=CSV_PATH.name,
        source_page="https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/",
        n_training_transitions=int(train.sum()),
        training_rule="transitions t->t+1 with t+1 <= 2020-11-30 (trailing 7-day means)",
        validation="recursion initialised at observed 2020-12-01 value; 2020-12-01 to 2021-04-06",
        gamma_hat=g, gamma_hat_v100_rule=g_old, turnover_days=1 / g,
        gamma_se_ols=float(ols.bse[0]), gamma_se_hac7=float(hac.bse[0]),
        gamma_ci95_ols=[g - 1.96 * float(ols.bse[0]), g + 1.96 * float(ols.bse[0])],
        gamma_ci95_hac=[g - 1.96 * float(hac.bse[0]), g + 1.96 * float(hac.bse[0])],
        gamma_ci95_block_bootstrap=[float(boot_ci[0]), float(boot_ci[1])],
        turnover_days_ci95_bootstrap=[float(1 / boot_ci[1]), float(1 / boot_ci[0])],
        gamma_fitted_on_validation_period=g_val_period,
        training_residual_ljungbox=lb_tr.to_dict(),
        validation_R2=r2, validation_RMSE=rmse, validation_MAE=mae,
        validation_band_from_gamma_ci=dict(R2_low=r2_rmse(obs, pred_lo)[0], R2_high=r2_rmse(obs, pred_hi)[0]),
        validation_one_step=dict(mean=float(one_step.mean()), sd=float(one_step.std(ddof=1)),
                                 acf1=acf1, acf7=acf7, ljungbox=lb_val.to_dict()),
        benchmarks=bench, prospective=prosp,
        rolling_gamma_28d_range=[float(min(r[1] for r in roll)), float(max(r[1] for r in roll))],
        peaks=peak_out,
        peak_raw_admissions=int(adm.max()), peak_raw_occupancy=int(occ.max()),
        final14_mean_occupancy=float(occ[-14:].mean()),
        peak_to_final14=float(occ.max() / occ[-14:].mean()),
    )
    (OUT / "england_empirical_results.json").write_text(json.dumps(res, indent=2, default=str))

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10.6, 7.4))
    a0 = ax[0, 0]
    a0.plot(d7, A7 / A7.max(), label="admissions + diagnoses (7-d mean)")
    a0.plot(d7, H7 / H7.max(), label="occupied beds (7-d mean)")
    a0.plot(dates, adm / adm.max(), lw=0.6, alpha=0.5, color="C0")
    a0.plot(dates, occ / occ.max(), lw=0.6, alpha=0.5, color="C1")
    for (da, do), st in zip([peaks["raw"], peaks["trailing7_end_labelled"]], [":", "--"]):
        a0.axvline(da, color="C0", ls=st, lw=0.8); a0.axvline(do, color="C1", ls=st, lw=0.8)
    a0.axvline(VAL_START, color="k", lw=0.8)
    a0.set_ylabel("normalised by own peak"); a0.set_title("(a) series; dotted: raw peaks, dashed: 7-d peaks")
    a0.legend(fontsize=7); a0.tick_params(axis="x", rotation=30)
    a1 = ax[0, 1]
    a1.plot(vdates, obs, label="observed", color="k")
    a1.plot(vdates, pred, "--", label=fr"stock balance, $\hat\gamma_H$={g:.4f}")
    a1.fill_between(vdates, pred_lo, pred_hi, alpha=0.25, label=r"95% bootstrap range of $\hat\gamma_H$")
    a1.plot(vdates, beta * S[s:], ":", label=f"{Lw}-day admissions sum")
    a1.plot(vdates, X[s:] @ coef, "-.", lw=0.9, label="distributed lag (21 d)")
    a1.set_ylabel("occupied COVID-19 beds"); a1.set_title("(b) held-out reconstruction and benchmarks")
    a1.legend(fontsize=7); a1.tick_params(axis="x", rotation=30)
    a2 = ax[1, 0]
    rd = [datetime.strptime(r[0], "%Y-%m-%d") for r in roll]
    rg = np.array([r[1] for r in roll]); rs = np.array([r[2] for r in roll])
    a2.plot(rd, rg); a2.fill_between(rd, rg - 1.96 * rs, rg + 1.96 * rs, alpha=0.25)
    a2.axhline(g, ls="--", color="k", lw=0.8); a2.axvline(VAL_START, color="k", lw=0.8)
    a2.set_ylabel(r"rolling 28-d $\hat\gamma_H$ (day$^{-1}$)"); a2.set_title("(c) constancy of turnover")
    a2.tick_params(axis="x", rotation=30)
    a3 = ax[1, 1]
    vals = obs - pred
    a3.plot(vdates, vals, label="recursive residual")
    a3.plot(vdates[1:], one_step, label="one-step residual")
    a3.axhline(0, color="k", lw=0.6); a3.set_ylabel("beds"); a3.set_title("(d) validation residuals")
    a3.legend(fontsize=7); a3.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_england_empirical.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "fig_england_empirical.png", dpi=200, bbox_inches="tight")
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
