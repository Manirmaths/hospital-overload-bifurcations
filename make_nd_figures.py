#!/usr/bin/env python3
"""Rebuild the global-bifurcation and atlas figures (release 1.1.0).

v1.0.0 drew the atlas with a b_trans search restricted to tau <= 100 (see
CHANGELOG.md).  This script now delegates to make_revision_figures.py, which
uses the certified b_trans, draws the collocation overlay and the fold curves,
and writes results/global_bifurcation_summary.json from the archived data.
"""
import json
from pathlib import Path

import make_revision_figures as mrf

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    mrf.fig_global()
    mrf.fig_atlas()
    fl = json.load(open(ROOT / "results" / "collocation_fold_refine_lower.json"))
    fu = json.load(open(ROOT / "results" / "collocation_fold_refine_upper.json"))
    fl = [x for x in fl if x["interior"]]; fu = [x for x in fu if x["interior"]]
    summary = dict(
        bH=2.32155208e-4, bcrit=1.0256571258926393e-3,
        bLPC_lower_shooting=8.6240659e-05, bLPC_upper_shooting=2.0924901645748015e-3,
        bLPC_lower_collocation=[x["b_fold"] for x in fl], bLPC_upper_collocation=[x["b_fold"] for x in fu],
        b_grazing_unstable_branch=1.0074e-4,
    )
    json.dump(summary, open(ROOT / "results" / "global_bifurcation_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
