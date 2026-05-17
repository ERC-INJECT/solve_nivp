"""Overlay slip profiles from the conformal vs SBM sweep against analytical."""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "_sweep_runs"

cases = []
for f in sorted(OUT_DIR.glob("mc_sliding_*.json")):
    d = json.load(open(f))
    cases.append(d)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

s_dense = np.linspace(-1, 1, 401)
for ax in (ax_l, ax_r):
    ax.plot(s_dense, np.sqrt(1 - s_dense**2), "k--", lw=1.4, label="analytical")

cmap_conformal = plt.cm.Blues
cmap_sbm = plt.cm.Reds
markers = {20: "o", 30: "s", 40: "^", 50: "D", 60: "v", 80: "P"}

conformal_Ns = sorted({d["N_ELEM"] for d in cases if not d["USE_SBM"]})
sbm_Ns = sorted({d["N_ELEM"] for d in cases if d["USE_SBM"]})

def shade(cmap, idx, total):
    return cmap(0.35 + 0.55 * idx / max(total - 1, 1))

for d in sorted(cases, key=lambda d: (d["USE_SBM"], d["N_ELEM"])):
    s_param = np.asarray(d["s_param_with_tips"])
    slip_phys = np.asarray(d["slip_with_tips_phys"])
    slip_norm = slip_phys / d["slip_max_anal"]
    N = d["N_ELEM"]
    if d["USE_SBM"]:
        c = shade(cmap_sbm, sbm_Ns.index(N), len(sbm_Ns))
        ax = ax_r
        label = f"SBM N={N}  (rel L2={d['rel_l2']:.3f})"
    else:
        c = shade(cmap_conformal, conformal_Ns.index(N), len(conformal_Ns))
        ax = ax_l
        label = f"conformal N={N}  (rel L2={d['rel_l2']:.3f})"
    ax.plot(s_param, slip_norm, marker=markers[N], color=c, ms=4.5, lw=1.0,
            label=label)

for ax, title in zip((ax_l, ax_r), ("Conformal", "SBM (USE_SBM=True, INCLUDE_SBM=False, INCLUDE_TAYLOR=True)")):
    ax.set_xlabel(r"$s = \xi / c$")
    ax.set_ylabel(r"$[\![u^{\rm t}]\!] \,/\, \frac{2(1-\nu)c\,\Delta\tau}{G}$")
    ax.set_title(title)
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(-1, color="0.6", lw=0.6)
    ax.axvline( 1, color="0.6", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
out_png = OUT_DIR / "mc_sliding_sweep_overlay.png"
fig.savefig(out_png, dpi=140)
print(f"wrote {out_png}")

# convergence-rate panel
fig2, ax = plt.subplots(figsize=(6, 4.5))
for mode, marker, color in (("conformal", "o", "#1f77b4"), ("sbm", "s", "#d62728")):
    pts = sorted([(d["N_ELEM"], d["rel_l2"]) for d in cases
                  if (d["USE_SBM"] == (mode == "sbm"))])
    Ns, errs = zip(*pts)
    ax.loglog(Ns, errs, marker=marker, color=color, lw=1.2, ms=8, label=mode)

# 1st-order reference
ref_x = np.array([20, 40], dtype=float)
ref_y = 0.4 * (20.0 / ref_x)
ax.loglog(ref_x, ref_y, "--", color="0.5", lw=1.0, label=r"$\propto 1/N$ (first order)")
ax.set_xlabel("N_ELEM")
ax.set_ylabel("Pollard relative L2 error  (interior |s|<0.95)")
ax.set_title("Convergence to Pollard-Segall analytical")
ax.grid(alpha=0.3, which="both")
ax.legend()

plt.tight_layout()
out_png2 = OUT_DIR / "mc_sliding_sweep_convergence.png"
fig2.savefig(out_png2, dpi=140)
print(f"wrote {out_png2}")
