# Hospital-overload bifurcations

Reproducible code, processed data, and numerical outputs supporting the manuscript

> **Hopf bifurcation, bistability and transient grazing in a piecewise-smooth hospital-overload model**
> (Scientific Reports, submission dc51fde4-f058-437c-a595-414061012e0d, revision 1)

**Release 1.1.0** is the frozen version associated with the revised manuscript.
Release 1.0.0 accompanied the original submission; see `CHANGELOG.md` for the
correction to the transient-threshold search and other changes.

## Archived releases

| Version | Use | DOI |
|---|---|---|
| 1.1.0 | revised manuscript | **10.5281/zenodo.NEWVERSION** (version DOI) |
| 1.0.0 | original submission | 10.5281/zenodo.21807502 |

## Environment

The archived results were produced with Python 3.13.17 and the exact package
versions in `requirements.txt` (numpy, scipy, matplotlib, sympy, mpmath,
statsmodels, jax[cpu]).

```bash
python -m pip install -r requirements.txt
```

## Run order

| Step | Command | Main outputs | Approx. CPU time |
|---|---|---|---|
| 1 | `python verify_revision.py` | `verification_results.json`, `fig_stability.pdf` | 2 min |
| 2 | `python independent_validation.py` | `results/independent_validation_results.json` | 5 min |
| 3 | `python revision_thresholds.py` | `results/threshold_atlas.json`, `results/ic_robustness.json` | 6 min |
| 4 | `python analysis_equilibria.py` | `results/capacity_equilibria_quadratic.json` | 2 min |
| 5 | `python grazing_diagnostics.py` | `results/grazing_diagnostics.json` | 10 min |
| 6 | `python collocation_folds.py 1e-6 -1` and `... 1e-6 1` | `results/collocation_branch_*.json` | 30-60 min each |
| 7 | `python collocation_refine_fold.py lower` / `upper` | `results/collocation_fold_refine_*.json` | 20 min each |
| 8 | `python saltation_check.py` | `results/saltation_check.json` | 5 min |
| 9 | `python bistability_demo.py 31` | `results/bistability.json` | 15 min |
| 10 | `python robustness.py oat` and `python robustness.py Bcurve` | `results/robustness_*.json` | 30 min each |
| 11 | `python dimensional_sensitivity.py` | `results/dimensional_sensitivity.json` | 1 min |
| 12 | `python england_empirical_analysis.py` | `results/england_empirical_results.json`, `fig_england_empirical.pdf` | <1 min |
| 13 | `python make_revision_figures.py` and `python make_nd_figures.py` | all manuscript figures | 5 min |

Steps 6-7 need the outputs of step 6 before step 7.

## What each check establishes

- **Transient threshold** (`thresholds.py`): b_trans is a supremum over an
  infinite horizon. It is computed on doubled horizons until a modal tail
  certificate shows that no later maximum can exceed the one found, and the
  ordering b_trans >= b_crit (Theorem 7.1) is asserted automatically.
- **Folds of periodic orbits**: primary single shooting (`fixed_b_cycle.py`,
  `amplitude_custom.py`, `qmax_continuation.py`), independent collocation
  pseudo-arclength continuation (`collocation_folds.py`,
  `collocation_refine_fold.py`), and hysteresis-simulation brackets
  (`robustness.py`). The methods agree to 1e-5 relative.
- **Switching-surface crossings**: the vector field is continuous, so the
  saltation matrix at transversal crossings is the identity;
  `saltation_check.py` verifies this numerically.

## Archived branch data (v1.0.0, unchanged)

- `unstable_branch_hopf.json`, `unstable_branch_hopf_tail.json`: unstable periodic branch from the subcritical Hopf point;
- `stable_branch_down.json`, `stable_branch_up.json`: attracting periodic branch;
- `qmax_branch_parts/qmax_branch_01.json` ... `04.json`: upper fold and return branch;
- `fold_refine.json`: lower-fold refinement.

## Data

`england_covid_hospital_2020-08-01_to_2021-04-06.csv` is the processed national
aggregate series used in the illustrative bed-turnover check. The underlying
public source is NHS England. No patient-level or personally identifiable data
are included.

## Licence

The code is released under the MIT License. The processed NHS England data remain subject to the terms of the original public source.

## Citation

Ibrahim, M., Alhassan, B., Ibrahim, S., Ibrahim, S. (2026). *Hospital-Overload Bifurcations:
Reproducible Code and Data for a Capacity-Limited Piecewise-Smooth Epidemic Model*
(Version 1.1.0). Zenodo. https://doi.org/10.5281/zenodo.NEWVERSION
