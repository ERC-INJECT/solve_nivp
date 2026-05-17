"""Decompose Pollard L2 error into interior vs near-tip contributions.

Loads slip profile JSONs from examples/_sweep_runs/ and computes:
  - residual r(s)  = | slip_FE_norm(s) - sqrt(1-s^2) |  evaluated at FE nodes
  - tip-decay log-log: r vs (1-|s|)  (Pollard tail)
  - interior  rel L2: |s| < 0.5
  - near-tip  rel L2: 0.5 <= |s| < 0.95
  - tip-edge  rel L2: 0.95 <= |s| < 1
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "_sweep_runs"


def load_cases():
    cases = []
    for f in sorted(OUT_DIR.glob("mc_sliding_*.json")):
        cases.append(json.load(open(f)))
    return cases


def banded_l2(s, r_abs, lo, hi):
    """relative L2 of residual r over the band lo <= |s| < hi, ref = analytical norm there."""
    mask = (np.abs(s) >= lo) & (np.abs(s) < hi)
    if not np.any(mask):
        return float("nan"), float("nan")
    ref = np.sqrt(np.clip(1.0 - s[mask] ** 2, 0.0, None))
    err = float(np.linalg.norm(r_abs[mask]))
    rfn = float(np.linalg.norm(ref))
    return err, err / max(rfn, 1e-30)


def analyse(d):
    s = np.asarray(d["s_param_with_tips"], dtype=float)
    slip_norm = np.asarray(d["slip_with_tips_phys"]) / d["slip_max_anal"]
    anal = np.sqrt(np.clip(1.0 - s**2, 0.0, None))
    r = np.abs(slip_norm) - np.abs(anal)        # signed (FE - analytical)
    r_abs = np.abs(r)
    return s, slip_norm, anal, r, r_abs


def main() -> int:
    cases = load_cases()

    print(f"{'mode':10s} {'N':>4s}  "
          f"{'rel L2 |s|<0.5':>14s}  "
          f"{'rel L2 0.5<|s|<0.95':>20s}  "
          f"{'rel L2 |s|>=0.95':>17s}")
    print("-" * 75)
    rows = []
    for d in sorted(cases, key=lambda d: (d["USE_SBM"], d["N_ELEM"])):
        s, slip_norm, anal, r, r_abs = analyse(d)
        _, rel_int  = banded_l2(s, r_abs, 0.0,  0.5)
        _, rel_tail = banded_l2(s, r_abs, 0.5,  0.95)
        _, rel_tip  = banded_l2(s, r_abs, 0.95, 1.0)
        mode = "sbm" if d["USE_SBM"] else "conformal"
        rows.append((mode, d["N_ELEM"], rel_int, rel_tail, rel_tip))
        print(f"{mode:10s} {d['N_ELEM']:4d}  "
              f"{rel_int:14.4f}  {rel_tail:20.4f}  {rel_tip:17.4f}")

    # ------- per-node residual along s, three biggest N's, both modes -------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    chosen = {40, 60, 80}
    cmap = {"conformal": plt.cm.Blues, "sbm": plt.cm.Reds}
    Ns_sorted = sorted(chosen)

    for ax, mode in zip(axes, ("conformal", "sbm")):
        for d in sorted(cases, key=lambda d: d["N_ELEM"]):
            if d["N_ELEM"] not in chosen:
                continue
            if (mode == "sbm") != d["USE_SBM"]:
                continue
            s, slip_norm, anal, r, _ = analyse(d)
            color = cmap[mode](0.4 + 0.5 * Ns_sorted.index(d["N_ELEM"]) / 2)
            ax.plot(s, r, "-o", color=color, ms=3.5, lw=1.0,
                    label=f"{mode} N={d['N_ELEM']} (rel L2={d['rel_l2']:.3f})")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel(r"$|u_{\rm num}| - |u_{\rm anal}|$")
        ax.set_title(f"signed slip residual — {mode}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[1].set_xlabel(r"$s = \xi / c$")
    plt.tight_layout()
    out1 = OUT_DIR / "mc_sliding_residuals_along_s.png"
    fig.savefig(out1, dpi=140)
    print(f"\nwrote {out1}")

    # ------- tip-decay log-log diagnostic -------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for d in sorted(cases, key=lambda d: (d["USE_SBM"], d["N_ELEM"])):
        if d["N_ELEM"] not in chosen:
            continue
        s, _, _, _, r_abs = analyse(d)
        td = 1.0 - np.abs(s)
        mask = (td > 0) & (r_abs > 0)
        mode = "sbm" if d["USE_SBM"] else "conformal"
        c = cmap[mode](0.4 + 0.5 * Ns_sorted.index(d["N_ELEM"]) / 2)
        marker = "o" if mode == "conformal" else "s"
        ax2.loglog(td[mask], r_abs[mask], marker=marker, ms=4.5, lw=0,
                   color=c, label=f"{mode} N={d['N_ELEM']}")
    x_ref = np.array([1e-2, 1.0])
    ax2.loglog(x_ref, 0.4 * x_ref**0.5, "--", color="0.4", lw=1.0, label=r"slope 1/2 ($P_1$)")
    ax2.loglog(x_ref, 0.4 * x_ref**1.5, ":",  color="0.4", lw=1.0, label=r"slope 3/2 ($P_2$)")
    ax2.set_xlabel(r"$1 - |s|$ (tip distance)")
    ax2.set_ylabel(r"$|u_{\rm num} - u_{\rm anal}|$")
    ax2.set_title("Tip-singularity decay: SBM vs conformal at N=40, 60, 80")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    out2 = OUT_DIR / "mc_sliding_residuals_tip_decay.png"
    fig2.savefig(out2, dpi=140)
    print(f"wrote {out2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
