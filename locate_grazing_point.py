"""Locate the grazing point b_g of the unstable periodic branch (min Psi = 0) by local collocation continuation."""
import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from collocation_refine_fold import refine, OUT
rows=json.load(open(OUT/'collocation_branch_tol1e-06_dir-1.json'))['rows']
r=refine(rows[18],1e-6,0.005,40,+1)
pts=[(x['b'],x['psimin']) for x in r['rows']]
for b,p in pts: print(f"{b:.6e} {p:.3e}")
for (b1,p1),(b2,p2) in zip(pts[:-1],pts[1:]):
    if p1<0<=p2: print('b_g', b1+(-p1)/(p2-p1)*(b2-b1))
