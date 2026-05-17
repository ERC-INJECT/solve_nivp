"""Build the SBM kinematic audit notebook from embedded_crack_mc_sliding.ipynb.

The audit notebook is the sliding test at SBM N=40 with the analytical
comparison cell wrapped by a sequence of audit cells that:

  1. Replicate the same converged state via the existing surrogate-frame
     extractor `D_s` (output stored as `sliding_slip[-1]`) — already done
     by the source notebook.
  2. Build a temporary `CrackInterfaceAssembler` and call
     `assemble_trace_u_plus_minus(include_taylor=True)` to obtain a
     true-crack-frame slip extractor.
  3. Apply both extractors to the converged bulk state and compare to
     Pollard-Segall.

The build script does not execute the notebook; the user opens it in
Jupyter to step through and inspect each transformation.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
SRC_NB = ROOT / "embedded_crack_mc_sliding.ipynb"
OUT_NB = ROOT / "sbm_kinematic_audit.ipynb"


# ----- cell helpers ------------------------------------------------------

def find_cell_index_with(nb, *substrings) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if all(s in c.source for s in substrings):
            return i
    raise RuntimeError(f"could not find cell with all of {substrings!r}")


def md(text: str):
    return nbformat.v4.new_markdown_cell(source=text)


def code(src: str):
    return nbformat.v4.new_code_cell(source=src)


# ----- override cells (force USE_SBM=True at N=40) -----------------------

PARAM_OVERRIDE_SRC = """\
# --- audit override: force SBM at N=40 ---
USE_SBM = True
CONFORMING_CRACK_MESH = not USE_SBM
INCLUDE_SBM = False
INCLUDE_TAYLOR = True
N_ELEM = 40
"""


# ----- audit cells -------------------------------------------------------

AUDIT_HEADER_MD = r"""\
# SBM kinematic audit

The sliding test above runs SBM at $N_{\rm ELEM}=40$ and compares the
finite-element slip against the Pollard–Segall analytical
$\sqrt{1-s^2}$ profile.  The slip values it plots come from the
**existing surrogate-frame extractor** `D_s`, defined as

```
D_s = T_u_s[jmpu_local_s, :].tocsr()
```

`T_u_s` is the algebraic ±½ avg/jmp transform composed with the per-node
rotation `R_v` into the **true-crack** $(n,t)$ frame.  It contains **no**
Taylor shift δ·∇u, so the values it returns are the discrete jump at
the **surrogate vertices**, not at the true crack.

The audit below builds the alternative extractor

```
[[û]] = (C_exp_u_plus + C_exp_u_minus) @ u_orig    with include_taylor=True
```

via the library's existing `assemble_trace_u_plus_minus` and applies the
same xy → $(n,t)$ rotation and component-major → interlaced permutation
the existing wiring uses, so the only mathematical difference between
the two operators is the Taylor shift.  We then apply both to the
converged bulk state and compare each against Pollard–Segall.

**Transformation chain** for the converged dynamic state $y$:

1. `y[u_state_indices]` → static block of u DOFs in **(n,t) rotated**
   avg/jmp layout (this is what `_apply_nodal_transform` produces when
   `rotated_to_nt=True`, line 2961 of `cgporoelastostatics.py`).
2. `_T_inv` (already includes the inverse of the (n,t) rotation,
   composed at line 2967) maps back to per-side `(u⁺, u⁻)` in **xy**
   at the surrogate vertices: `u_orig = _T_inv[Np:Np+Nu, :] @ y_static`.
3. Library trace operator: `[[û]]_xy = (C_exp_u_plus + C_exp_u_minus) @ u_orig`,
   with the Taylor shift δ·∇u applied if `include_taylor=True`.
   Output layout: component-major xy at $n_u$ interface nodes.
4. Per-node rotation xy → $(n,t)$ using `interface_normals` (which point
   to the **true** crack normal — `_compute_interface_normals` projects
   each surrogate node onto the true crack and reads the analytical
   tangent there, line 1533 of `cg_crack_assembler.py`).
5. Permute component-major → interlaced $(n_0,t_0,n_1,t_1,\ldots)$ to
   match the layout the contact solver and the existing diagnostics use.
"""

AUDIT_BUILD_SRC = """\
# Audit cell A1: build temporary assembler and trace operators
from poroelasticity.cg_crack_assembler import (
    CrackInterfaceAssembler, CrackAssemblyContext,
)

_ctx_audit = CrackAssemblyContext.from_solver_for_crack(poro_s, 0)
_asm_audit = CrackInterfaceAssembler(
    _ctx_audit,
    enforcement_type=poro_s.enforcement_type,
    taylor_method=poro_s.taylor_method,
    crack_law=poro_s.crack_law,
    tangential_flow=getattr(poro_s.crack_models[0], 'tangential_flow', False),
    fracture_perm=getattr(poro_s._ndp_per_crack[0], 'fracture_perm_d', 0.0),
    reference_aperture=getattr(poro_s._ndp_per_crack[0], 'reference_aperture_d', 0.0),
)
_asm_audit.assemble_all(
    include_hessian=False,
    include_taylor=poro_s.include_taylor,
    include_taylor_test=False,
    include_sbm=False,
    lumped_coupling='consistent',
)

C_exp_plus_with, C_exp_minus_with = _asm_audit.assemble_trace_u_plus_minus(
    include_taylor=True)
C_exp_plus_no,   C_exp_minus_no   = _asm_audit.assemble_trace_u_plus_minus(
    include_taylor=False)

C_jmp_with = (C_exp_plus_with + C_exp_minus_with).tocsr()
C_jmp_no   = (C_exp_plus_no   + C_exp_minus_no  ).tocsr()

n_u_nodes_audit = poro_s.interface_normals.shape[0]
print(f"interface nodes: n_u_nodes = {n_u_nodes_audit}, n_c (contact) = {n_c_s}")
assert n_u_nodes_audit == n_c_s, (
    f"audit assumes interface_normals length matches n_c; got "
    f"{n_u_nodes_audit} vs {n_c_s}"
)
print(f"C_jmp_with.shape = {C_jmp_with.shape}, C_jmp_no.shape = {C_jmp_no.shape}")
print(f"||C_with - C_no||_F / ||C_no||_F = "
      f"{(sp.linalg.norm(C_jmp_with - C_jmp_no) / sp.linalg.norm(C_jmp_no)):.4e}")
"""

AUDIT_ROTPERM_SRC = """\
# Audit cell A2: per-node xy -> (n,t) rotation + component-major -> interlaced
n_au = n_u_nodes_audit
rows_R, cols_R, data_R = [], [], []
for k in range(n_au):
    nx, ny = poro_s.interface_normals[k]
    # row 'n' at node k -> output row k (component-major: 0*n + k)
    rows_R += [k, k]
    cols_R += [0 * n_au + k, 1 * n_au + k]
    data_R += [nx, ny]
    # row 't' at node k -> output row n + k (component-major: 1*n + k)
    rows_R += [n_au + k, n_au + k]
    cols_R += [0 * n_au + k, 1 * n_au + k]
    data_R += [-ny, nx]
R_xy_to_nt = sp.csr_matrix(
    (data_R, (rows_R, cols_R)), shape=(2 * n_au, 2 * n_au))

# permutation: (n_0,n_1,...,t_0,t_1,...) -> (n_0,t_0,n_1,t_1,...)
perm_idx = [r for k in range(n_au) for r in (k, n_au + k)]
P_cm_to_il = sp.csr_matrix(
    (np.ones(len(perm_idx)), (np.arange(len(perm_idx)), perm_idx)),
    shape=(len(perm_idx), len(perm_idx)),
)

D_audit_no_taylor   = (P_cm_to_il @ R_xy_to_nt @ C_jmp_no  ).tocsr()
D_audit_with_taylor = (P_cm_to_il @ R_xy_to_nt @ C_jmp_with).tocsr()
print(f"D_audit_no_taylor.shape   = {D_audit_no_taylor.shape}")
print(f"D_audit_with_taylor.shape = {D_audit_with_taylor.shape}")
"""

AUDIT_INTERLOCK_SRC = """\
# Audit cell A3: INTERLOCK
# Recover original-frame per-side u from the converged static block.
# poro_s._T_inv already includes the (n,t)->xy unrotation (line 2967 of
# cgporoelastostatics.py composes T_inv = T_inv @ R_block_T), so a single
# slice of the inverse transform suffices.
y_final_audit = np.asarray(y_sliding[-1], dtype=float)
y_static_full = y_final_audit[:n_base_s]
T_inv_u_block = poro_s._T_inv[Np_s:Np_s + Nu_s, :].tocsr()
u_orig = np.asarray(T_inv_u_block @ y_static_full, dtype=float).ravel()
print(f"u_orig.shape = {u_orig.shape}, |u_orig|_max = {np.abs(u_orig).max():.3e}")

slip_audit_no_taylor   = np.asarray(D_audit_no_taylor   @ u_orig, dtype=float)
slip_audit_with_taylor = np.asarray(D_audit_with_taylor @ u_orig, dtype=float)

# Existing extractor returns the tangential jumps directly (n_c values).
# Our audit extractors return interlaced (n,t) (2*n_c values).
slip_existing_t = np.asarray(sliding_slip[-1], dtype=float)
slip_audit_no_t = slip_audit_no_taylor[1::2]   # tangential at each node
slip_audit_no_n = slip_audit_no_taylor[0::2]   # normal jump

err_t = np.linalg.norm(slip_audit_no_t - slip_existing_t)
ref_t = max(np.linalg.norm(slip_existing_t), 1e-30)
print(f"INTERLOCK: ||audit_no_taylor (tangential) - sliding_slip[-1]|| / ||sliding_slip[-1]|| "
      f"= {err_t / ref_t:.3e}")
print(f"  (should be near machine epsilon if frame conventions match)")
print(f"audit_no_taylor max |normal-jump|   = {np.abs(slip_audit_no_n).max():.3e}")
print(f"  (expected near 0 for a sliding test that maintains normal contact)")
"""

AUDIT_COMPARE_SRC = """\
# Audit cell A4: compare both extractors against Pollard-Segall
slip_t_no_taylor   = slip_audit_no_taylor[1::2]
slip_t_with_taylor = slip_audit_with_taylor[1::2]
slip_n_no_taylor   = slip_audit_no_taylor[0::2]
slip_n_with_taylor = slip_audit_with_taylor[0::2]

# Convert to physical and apply the same arclength sort the source uses.
slip_phys_no   = slip_t_no_taylor   * DISP_SCALE
slip_phys_with = slip_t_with_taylor * DISP_SCALE

slip_phys_no_sorted   = slip_phys_no  [_sort_order]
slip_phys_with_sorted = slip_phys_with[_sort_order]

# Augment with the geometric-tip zeros (same protocol as cell 16)
slip_no_with_tips   = np.concatenate(([0.0], slip_phys_no_sorted,   [0.0]))
slip_with_with_tips = np.concatenate(([0.0], slip_phys_with_sorted, [0.0]))

slip_anal_phys = slip_max_anal * np.sqrt(np.clip(1.0 - s_param**2, 0.0, None))

mask = np.abs(s_param) < 0.95
err_no   = float(np.linalg.norm(np.abs(slip_phys_no_sorted  [mask]) - np.abs(slip_anal_phys[mask])))
err_with = float(np.linalg.norm(np.abs(slip_phys_with_sorted[mask]) - np.abs(slip_anal_phys[mask])))
ref      = float(np.linalg.norm(np.abs(slip_anal_phys[mask])))

print(f"\\nPollard rel L2 (interior |s|<0.95):")
print(f"  surrogate-frame  (no  Taylor): {err_no   / ref:.4f}")
print(f"  true-crack-frame (with Taylor): {err_with / ref:.4f}")
print(f"  (existing notebook reported  : {err_l2 / ref:.4f})")

# Also report the change in slip magnitude at the centre
i_centre = int(np.argmin(np.abs(s_param)))
print(f"\\nSlip at s ~ 0 (center of crack):")
print(f"  no-Taylor   centre slip = {slip_phys_no_sorted[i_centre]:.4e}")
print(f"  with-Taylor centre slip = {slip_phys_with_sorted[i_centre]:.4e}")
print(f"  analytical  centre slip = {slip_max_anal:.4e}")
"""

AUDIT_DEEP_SRC = """\
# Audit cell A4b: deeper diagnostic — is Taylor *actually* contributing?
# ---------------------------------------------------------------------
# Compute the pure Taylor-correction matrix and apply to u_orig directly.
C_taylor_corr_xy = (C_jmp_with - C_jmp_no).tocsr()
print(f"||C_jmp_with - C_jmp_no||_F                = "
      f"{sp.linalg.norm(C_taylor_corr_xy):.4e}")
print(f"||C_jmp_with - C_jmp_no||_F / ||C_jmp_no||_F = "
      f"{sp.linalg.norm(C_taylor_corr_xy) / sp.linalg.norm(C_jmp_no):.4e}")

slip_taylor_correction_xy = np.asarray(C_taylor_corr_xy @ u_orig, dtype=float)
print(f"max |slip_taylor_correction_xy| (component-major xy) = "
      f"{np.abs(slip_taylor_correction_xy).max():.3e}")

# Apply same rotation+permutation as the main extractors
slip_taylor_correction_nt = np.asarray(
    (P_cm_to_il @ R_xy_to_nt @ C_taylor_corr_xy) @ u_orig, dtype=float)
slip_corr_n = slip_taylor_correction_nt[0::2]
slip_corr_t = slip_taylor_correction_nt[1::2]
print(f"\\nTaylor correction in (n,t) layout, per node:")
print(f"  max |normal-correction|     = {np.abs(slip_corr_n).max():.3e}")
print(f"  max |tangential-correction| = {np.abs(slip_corr_t).max():.3e}")
print(f"  ||correction||_2 (interlaced) = "
      f"{np.linalg.norm(slip_taylor_correction_nt):.3e}")
print(f"  ||sliding_slip[-1]||_2        = "
      f"{np.linalg.norm(np.asarray(sliding_slip[-1], dtype=float)):.3e}")

# Probe delta directly
delta_plus_arr, delta_minus_arr = _asm_audit._compute_interface_delta()
delta_plus_norms = np.linalg.norm(delta_plus_arr, axis=1)
print(f"\\nSurrogate->true-crack offsets (delta) per interface node:")
print(f"  |delta|_min = {delta_plus_norms.min():.4e}")
print(f"  |delta|_max = {delta_plus_norms.max():.4e}")
print(f"  |delta|_mean = {delta_plus_norms.mean():.4e}")
# h is roughly 1/N for the unit box
h_est = 1.0 / N_ELEM
print(f"  reference h ~ 1/N_ELEM = {h_est:.4e}  "
      f"(|delta| should be O(h/2) for SBM, 0 for conformal)")
"""

AUDIT_PLOT_SRC = """\
# Audit cell A5: overlay plot
fig_audit, ax_audit = plt.subplots(figsize=(10, 5))
s_dense = np.linspace(-1, 1, 401)
ax_audit.plot(s_dense, np.sqrt(1.0 - s_dense ** 2), 'k--', lw=1.5,
              label=r'analytical $\\sqrt{1-s^2}$')
ax_audit.plot(s_dense, -np.sqrt(1.0 - s_dense ** 2), 'k--', lw=1.5, alpha=0.35)

ax_audit.plot(s_param,  slip_phys_no_sorted   / slip_max_anal,
              'ro-', ms=4, lw=1.0,
              label=f'FE no-Taylor (rel L2 = {err_no/ref:.3f})')
ax_audit.plot(s_param,  slip_phys_with_sorted / slip_max_anal,
              'bs-', ms=4, lw=1.0,
              label=f'FE Taylor-corrected (rel L2 = {err_with/ref:.3f})')

ax_audit.set_xlabel(r'$s = \\xi / c$')
ax_audit.set_ylabel(
    r'$[\\![u^t]\\!] \\,/\\, \\frac{2(1-\\nu)c\\,\\Delta\\tau}{G}$')
ax_audit.set_title(f'SBM N={N_ELEM} (theta_crack={np.degrees(CRACK_THETA_SLIDING):.1f} deg): '
                   'true-crack vs surrogate-frame slip')
ax_audit.set_xlim(-1.05, 1.05)
ax_audit.axvline(-1, color='0.6', lw=0.6); ax_audit.axvline(1, color='0.6', lw=0.6)
ax_audit.grid(alpha=0.3)
ax_audit.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()
"""


# ----- builder ------------------------------------------------------------

def build():
    nb = nbformat.read(SRC_NB, as_version=4)

    # 1. Override params after the user-inputs cell.
    param_idx = find_cell_index_with(nb, "USE_SBM", "N_ELEM", "INCLUDE_SBM")
    nb.cells.insert(param_idx + 1, code(PARAM_OVERRIDE_SRC))

    # 2. Insert audit cells after the analytical-comparison cell (cell that
    #    computes Pollard L2 — search for that block).
    pollard_idx = find_cell_index_with(nb, "Pollard L2", "slip_max_anal")
    audit_cells = [
        md(AUDIT_HEADER_MD),
        code(AUDIT_BUILD_SRC),
        code(AUDIT_ROTPERM_SRC),
        code(AUDIT_INTERLOCK_SRC),
        code(AUDIT_COMPARE_SRC),
        code(AUDIT_DEEP_SRC),
        code(AUDIT_PLOT_SRC),
    ]
    for j, c in enumerate(audit_cells):
        nb.cells.insert(pollard_idx + 1 + j, c)

    nbformat.write(nb, OUT_NB)
    print(f"wrote {OUT_NB}")


if __name__ == "__main__":
    build()
