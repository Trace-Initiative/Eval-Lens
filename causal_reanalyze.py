#!/usr/bin/env python3
"""
Causal reanalysis (CPU, no GPU) — reads the saved causal_f5.json and evaluates the F5 steering effect
with better statistics than the endpoint swing:
  (1) monotonic SLOPE of p(eval) vs alpha (F5 should be a positive outlier vs the random null)
  (2) MONOTONICITY (fraction of adjacent steps increasing) — F5 should be ~perfect, random noisy
  (3) restricted to the SMALL-alpha (in-distribution) regime, where random disruption is controlled

  python causal_reanalyze.py
"""
import os
import json
import numpy as np

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "causal")
res = json.load(open(os.path.join(OUTD, "causal_f5.json")))
alphas = np.array(res["alpha"], dtype=float)
f5 = np.array(res["F5"], dtype=float)
plc = np.array(res["placebo_curves"], dtype=float)   # [K, n_alpha]
K = plc.shape[0]


def slope(a, y):
    return float(np.polyfit(a, y, 1)[0])


def mono(y):
    d = np.diff(y)
    return float((d > 0).mean())   # fraction of steps that increase


for label, amax in [("full |a|<=2", 2.0), ("inner |a|<=1", 1.0), ("inner |a|<=0.5", 0.5)]:
    m = np.abs(alphas) <= amax + 1e-9
    a = alphas[m]
    s_f5 = slope(a, f5[m]); mo_f5 = mono(f5[m])
    s_pl = np.array([slope(a, plc[k][m]) for k in range(K)])
    mo_pl = np.array([mono(plc[k][m]) for k in range(K)])
    p_slope = (int((s_pl >= s_f5).sum()) + 1) / (K + 1)
    print(f"\n[{label}]  ({m.sum()} alpha points)")
    print(f"  F5:     slope={s_f5:+.3f}   monotonicity={mo_f5:.0%}")
    print(f"  random: slope median={np.median(s_pl):+.3f}  range[{s_pl.min():+.3f},{s_pl.max():+.3f}]  "
          f"monotonicity median={np.median(mo_pl):.0%}")
    print(f"  p-like (random slope >= F5) = {p_slope:.2f}   correctly-signed={s_f5>0}")
    print(f"  -> {'CAUSAL (positive slope outlier)' if s_f5 > 0 and p_slope <= 0.1 else 'suggestive/near-null'}")

print("\nRead: F5 correctly-signed + strongly monotonic + random mean-slope ~0 = real (if F5 slope is")
print("an outlier at small alpha). Report the honest verdict; do not overclaim.")
