# Changelog

## 1.1.0 (revision for Scientific Reports, October 2026)

### Corrected
- **Transient threshold search (`transient_threshold`).** v1.0.0 searched for the
  maximum of the admission pressure only on tau in [0, 100], although
  b_trans is defined as a supremum over [0, infinity). For transmission
  parameters near the invasion threshold (B <= 1.14, R0 <= 1.19) this gave
  b_trans < b_crit, contradicting Theorem 7.1 (e.g. B = 1.13889: v1.0.0
  7.24e-5, correct value 3.453e-3 at tau = 189.74), and it underestimated
  b_trans for B up to 1.32. Values for B >= 1.38, including the baseline
  B = 3, were correct to 1e-6. The new `thresholds.py` integrates on doubled
  horizons until a modal tail certificate shows that no later maximum can
  exceed the one found, and asserts b_trans >= b_crit. The old function is
  kept as `transient_threshold_v100` for documentation only.
- **Empirical train/validation boundary.** The final training transition
  (30 Nov -> 1 Dec) used the 1 Dec occupancy that also initialises
  validation. Training now uses only transitions whose end date is on or
  before 30 Nov (gamma_H: 0.09841 -> 0.09844 per day).
- Two-parameter atlas regenerated with the certified b_trans.

### Added
- `collocation_folds.py`, `collocation_refine_fold.py`: independent
  collocation + pseudo-arclength continuation of the periodic branches and
  folds (log coordinates, scipy `solve_bvp`).
- `saltation_check.py`: monodromy matrices by (a) pointwise switching,
  (b) event location with explicit saltation matrices, (c) finite
  differences.
- `bistability_demo.py`: trajectories and basin slices at b = 1.5e-4, 1.5e-3.
- `robustness.py`: +/-25 % one-at-a-time variations and B-sweep; Hopf
  criticality, fold brackets by hysteresis simulation, b_trans/b_crit.
- `revision_thresholds.py`: certified atlas and initial-condition study.
- `grazing_diagnostics.py`: event-located convergence diagnostics for the
  grazing laws, including exit-time shift and uniform deviation.
- `analysis_equilibria.py`: explicit quadratic for capacity-limited
  equilibria, symbolic verification, random parameter scan.
- `dimensional_sensitivity.py`: elasticities with respect to independent
  dimensional rates.
- `england_empirical_analysis.py`: HAC and block-bootstrap uncertainty,
  residual diagnostics, benchmarks, prospective forecasts, rolling gamma,
  raw versus smoothed peak timing.
- `make_revision_figures.py`: all revised figures.

### Documentation
- `requirements.txt` pins the exact environment and now includes JAX and
  statsmodels.
- README updated with the archived DOI and the run order.
