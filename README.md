# Hospital-overload bifurcations

Reproducible code, processed data, and numerical outputs supporting the manuscript:

> **Hopf bifurcation, bistability and transient grazing in a piecewise-smooth hospital-overload model**

The model represents capacity-limited hospital admission through a continuous piecewise-smooth switching law. This repository contains the calculations used for equilibrium analysis, local stability, Hopf bifurcation, first-Lyapunov-coefficient evaluation, periodic-orbit continuation, Floquet multipliers, grazing asymptotics, switching-time bounds, sensitivity analysis, and an NHS England hospital-stock illustration.

## Main reproducibility checks

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python verify_revision.py
python independent_validation.py
python england_empirical_analysis.py
```

- `verify_revision.py` runs the primary analytical and numerical checks.
- `independent_validation.py` independently reproduces the Hopf point, first Lyapunov coefficient, and one representative attracting cycle.
- `england_empirical_analysis.py` reproduces the hospital stock-flow illustration.
- `make_nd_figures.py` rebuilds the global bifurcation and two-parameter atlas figures from the archived branch data.

## Global periodic-orbit continuation

The archived branch files used in the manuscript are:

- `unstable_branch_hopf.json` and `unstable_branch_hopf_tail.json`: unstable periodic branch issuing from the subcritical Hopf point;
- `stable_branch_down.json` and `stable_branch_up.json`: attracting periodic branch;
- `qmax_branch_parts/qmax_branch_01.json` through `qmax_branch_04.json`: partitioned upper return branch and upper-fold data, loaded automatically by `make_nd_figures.py`;
- `fold_refine.json`: lower-fold refinement;
- `results/global_bifurcation_summary.json`: principal numerical landmarks.

The periodic branch was computed by single shooting with variational equations and Floquet multipliers. Overlapping continuation parameterisations are implemented in `fixed_b_cycle.py`, `amplitude_custom.py`, and `qmax_continuation.py`; `explore_continuation.py` constructs a small-amplitude Hopf seed.

## Numerical status and scope

The Hopf location, frequency, transversality, first Lyapunov coefficient, and one representative attracting cycle were reproduced independently. The two fold locations are primary single-shooting results supported by small periodicity residuals, overlapping parameterisations, and Floquet-multiplier crossings. They have not been independently collocated.

## Data

`england_covid_hospital_2020-08-01_to_2021-04-06.csv` is the processed national aggregate series used in the empirical stock-flow check. The underlying public source is NHS England and is cited in the manuscript. No patient-level or personally identifiable data are included.

## Licence

The code is released under the MIT License. The processed NHS England data remain subject to the terms of the original public source.

## Citation

A permanent archived release and DOI will be added through Zenodo. Until then, cite the associated manuscript and the tagged GitHub release used for your analysis.
