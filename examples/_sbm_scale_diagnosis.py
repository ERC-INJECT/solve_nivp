"""Diagnose whether SBM bulk slip bias is a pure scale factor.

For each N, fit the best constant alpha so that alpha * slip_sbm(s) matches
the analytical slip in the interior |s|<0.5 (where the slip profile is
near-quadratic in s and tip effects don't dominate).

If alpha != 1 by ~10% AND the residual after scaling drops to conformal-mesh
levels, then the SBM bias is dominantly a scale factor on the kinematic
coupling.  If alpha ~= 1 OR residual stays high, the bias has structure.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "_sweep_runs"


def load():
    cs = []
    for f in sorted(OUT_DIR.glob("mc_sliding_*.json")):
        cs.append(json.load(open(f)))
    return cs


def fit_scale_interior(d, smax=0.5):
    """Best alpha s.t. alpha * slip_norm_FE(s) ~ sqrt(1 - s^2) on |s|<smax."""
    s = np.asarray(d["s_param_with_tips"], dtype=float)
    slip_norm = np.asarray(d["slip_with_tips_phys"]) / d["slip_max_anal"]
    anal = np.sqrt(np.clip(1.0 - s ** 2, 0.0, None))
    mask = np.abs(s) < smax
    fe  = slip_norm[mask]
    an  = anal[mask]
    # least-squares scalar: alpha = (an . fe) / (fe . fe)
    alpha = float(np.dot(an, fe) / max(np.dot(fe, fe), 1e-30))

    # post-fit residual rel L2 over interior band
    fe_scaled = alpha * fe
    err = float(np.linalg.norm(fe_scaled - an))
    ref = float(np.linalg.norm(an))
    rel_resid = err / max(ref, 1e-30)

    # for comparison, pre-fit interior rel L2
    err0 = float(np.linalg.norm(fe - an))
    rel_pre = err0 / max(ref, 1e-30)

    return alpha, rel_pre, rel_resid


def main() -> int:
    cases = load()
    print(f"interior band |s| < 0.5\n")
    print(f"{'mode':10s} {'N':>4s}  {'alpha (best)':>12s}  "
          f"{'rel L2 pre-fit':>15s}  {'rel L2 post-fit':>16s}  "
          f"{'reduction':>10s}")
    print("-" * 78)
    for d in sorted(cases, key=lambda d: (d["USE_SBM"], d["N_ELEM"])):
        a, rpre, rpost = fit_scale_interior(d)
        mode = "sbm" if d["USE_SBM"] else "conformal"
        red = rpre / max(rpost, 1e-30)
        print(f"{mode:10s} {d['N_ELEM']:4d}  {a:12.4f}  "
              f"{rpre:15.4f}  {rpost:16.4f}  {red:10.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
