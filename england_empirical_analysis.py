#!/usr/bin/env python3
"""Reproduce the NHS England hospital-stock consistency check.

Input
-----
england_covid_hospital_2020-08-01_to_2021-04-06.csv

Outputs
-------
england_empirical_results.json
fig_england_empirical.pdf
fig_england_empirical.png

The reported admissions/diagnoses series is used as a proxy for entries into
recorded COVID-19 occupancy. This is a structural stock-flow check, not a full
calibration of the epidemic model or its capacity thresholds.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
CSV_PATH = OUT / "england_covid_hospital_2020-08-01_to_2021-04-06.csv"


def moving_average_valid(x: np.ndarray, window: int = 7) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main() -> None:
    dates, admissions, occupancy = [], [], []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            admissions.append(float(row["total_reported_admissions_and_diagnoses"]))
            occupancy.append(float(row["occupied_covid_beds_0800"]))

    dates = np.array(dates, dtype=object)
    admissions = np.asarray(admissions, dtype=float)
    occupancy = np.asarray(occupancy, dtype=float)

    adm7 = moving_average_valid(admissions, 7)
    occ7 = moving_average_valid(occupancy, 7)
    dates7 = dates[6:]

    # H_{t+1} - H_t = A_t - gamma H_t + error.
    # Fit through 30 Nov 2020, using seven-day means and no intercept.
    delta_h = occ7[1:] - occ7[:-1]
    a_t = adm7[:-1]
    h_t = occ7[:-1]
    balance_dates = dates7[:-1]
    train = np.array([d <= datetime(2020, 11, 30) for d in balance_dates])
    response = a_t - delta_h
    gamma_hat = float(np.dot(h_t[train], response[train]) / np.dot(h_t[train], h_t[train]))

    # Held-out recursive prediction from 1 Dec 2020.
    start_idx = int(np.where(dates7 == datetime(2020, 12, 1))[0][0])
    pred = np.empty(len(dates7) - start_idx)
    pred[0] = occ7[start_idx]
    for k in range(len(pred) - 1):
        j = start_idx + k
        pred[k + 1] = pred[k] + adm7[j] - gamma_hat * pred[k]
    obs = occ7[start_idx:]
    residual = obs - pred
    r2 = float(1.0 - np.sum(residual**2) / np.sum((obs - obs.mean())**2))
    rmse = float(np.sqrt(np.mean(residual**2)))

    peak_adm_idx = int(np.argmax(admissions))
    peak_occ_idx = int(np.argmax(occupancy))
    lag = int((dates[peak_occ_idx] - dates[peak_adm_idx]).days)
    final14 = float(np.mean(occupancy[-14:]))

    results = {
        "source_file": CSV_PATH.name,
        "source_page": "https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/",
        "period_start": dates[0].strftime("%Y-%m-%d"),
        "period_end": dates[-1].strftime("%Y-%m-%d"),
        "peak_daily_admissions_and_diagnoses": int(admissions[peak_adm_idx]),
        "peak_admissions_date": dates[peak_adm_idx].strftime("%Y-%m-%d"),
        "peak_occupied_beds": int(occupancy[peak_occ_idx]),
        "peak_occupancy_date": dates[peak_occ_idx].strftime("%Y-%m-%d"),
        "peak_lag_days": lag,
        "final_14_day_mean_occupancy": final14,
        "peak_to_final14_mean_ratio": float(occupancy[peak_occ_idx] / final14),
        "training_period": "2020-08-07 to 2020-11-30 (7-day means)",
        "validation_period": "2020-12-01 to 2021-04-06 (7-day means)",
        "effective_turnover_rate_per_day": gamma_hat,
        "effective_turnover_time_days": float(1.0 / gamma_hat),
        "validation_R2_occupancy_level": r2,
        "validation_RMSE_beds": rmse,
    }
    (OUT / "england_empirical_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].plot(dates7, adm7 / np.max(adm7), label="admissions/diagnoses")
    axes[0].plot(dates7, occ7 / np.max(occ7), label="occupied beds")
    axes[0].set_ylabel("normalised seven-day mean")
    axes[0].set_title("(a) transient wave and occupancy lag")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    val_dates = dates7[start_idx:]
    axes[1].plot(val_dates, obs, label="observed")
    axes[1].plot(val_dates, pred, linestyle="--", label="stock-balance prediction")
    axes[1].set_ylabel("occupied COVID-19 beds")
    axes[1].set_title("(b) held-out occupancy")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(OUT / "fig_england_empirical.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_england_empirical.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
