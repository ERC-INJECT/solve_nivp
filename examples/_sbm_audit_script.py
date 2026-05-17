"""Standalone SBM kinematic audit.

Builds the SBM N=40 sliding test by executing the relevant cells from
``embedded_crack_mc_sliding.ipynb`` in-process (so we share kernel state),
then runs the audit:

  * extractor without Taylor (must match existing ``D_s``)
  * extractor with    Taylor
  * applies both to the converged bulk state
  * reports per-node Taylor correction magnitude, max |delta|, and
    Pollard rel L2 with and without Taylor.

Usage:
    python examples/_sbm_audit_script.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parent
SRC_NB = ROOT / "embedded_crack_mc_sliding.ipynb"


PARAM_OVERRIDE_SRC = """\
USE_SBM = True
CONFORMING_CRACK_MESH = not USE_SBM
INCLUDE_SBM = False
INCLUDE_TAYLOR = True
N_ELEM = 40
"""


def first_idx_after(nb, *substrings):
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and all(s in c.source for s in substrings):
            return i
    raise RuntimeError(f"missing cell with all of {substrings}")


def main() -> int:
    print("loading notebook source...", flush=True)
    nb = nbformat.read(SRC_NB, as_version=4)

    # We want everything up to and including the analytical comparison cell
    # (the cell printing 'Pollard L2'), so we have the converged state
    # `y_sliding`, the contact wiring (`gap_extract_dyn_s`, etc.), and the
    # diagnostics (`s_param`, `slip_max_anal`, `_sort_order`, ...).
    last_idx = first_idx_after(nb, "Pollard L2", "slip_max_anal")
    param_idx = first_idx_after(nb, "USE_SBM", "N_ELEM", "INCLUDE_SBM")

    setup_cells = []
    for i, c in enumerate(nb.cells[: last_idx + 1]):
        if c.cell_type != "code":
            continue
        if i == param_idx:
            setup_cells.append(c.source)
            setup_cells.append(PARAM_OVERRIDE_SRC)
        else:
            setup_cells.append(c.source)

    print(f"setup cells: {len(setup_cells)}", flush=True)
    print("executing setup...", flush=True)
    t0 = time.time()
    ns = {"__name__": "__main__"}
    for j, src in enumerate(setup_cells):
        try:
            exec(compile(src, f"<cell_{j}>", "exec"), ns)
        except Exception as e:  # pragma: no cover -- diagnostics
            print(f"\n[cell {j}] FAILED: {type(e).__name__}: {e}", flush=True)
            print(f"first 200 chars of cell:\n{src[:200]}", flush=True)
            raise
    print(f"setup done in {time.time() - t0:.1f}s", flush=True)

    # Pull the bindings we need from the executed namespace
    poro_s = ns["poro_s"]
    y_sliding = ns["y_sliding"]
    n_base_s = ns["n_base_s"]
    n_c_s = ns["n_c_s"]
    Np_s = ns["Np_s"]
    Nu_s = ns["Nu_s"]
    sliding_slip_history = ns["sliding_slip"]
    DISP_SCALE = ns["DISP_SCALE"]
    s_param = ns["s_param"]
    _sort_order = ns["_sort_order"]
    slip_max_anal = ns["slip_max_anal"]
    err_l2 = ns["err_l2"]
    ref_l2 = ns["ref_l2"]
    N_ELEM = ns["N_ELEM"]
    USE_SBM = ns["USE_SBM"]

    print("\n=== AUDIT ===\n", flush=True)
    print(f"USE_SBM = {USE_SBM}, N_ELEM = {N_ELEM}", flush=True)
    print(f"existing notebook reported Pollard rel L2 = {err_l2 / ref_l2:.4f}", flush=True)

    # --- build temporary CrackInterfaceAssembler for poro_s ----------------
    from poroelasticity.cg_crack_assembler import (
        CrackInterfaceAssembler, CrackAssemblyContext,
    )

    ctx_audit = CrackAssemblyContext.from_solver_for_crack(poro_s, 0)
    asm_audit = CrackInterfaceAssembler(
        ctx_audit,
        enforcement_type=poro_s.enforcement_type,
        taylor_method=poro_s.taylor_method,
        crack_law=poro_s.crack_law,
        tangential_flow=getattr(poro_s.crack_models[0], 'tangential_flow', False),
        fracture_perm=getattr(poro_s._ndp_per_crack[0], 'fracture_perm_d', 0.0),
        reference_aperture=getattr(poro_s._ndp_per_crack[0], 'reference_aperture_d', 0.0),
    )
    asm_audit.assemble_all(
        include_hessian=False,
        include_taylor=poro_s.include_taylor,
        include_taylor_test=False,
        include_sbm=False,
        lumped_coupling='consistent',
    )

    Cp_with, Cm_with = asm_audit.assemble_trace_u_plus_minus(include_taylor=True)
    Cp_no,   Cm_no   = asm_audit.assemble_trace_u_plus_minus(include_taylor=False)
    C_jmp_with = (Cp_with + Cm_with).tocsr()
    C_jmp_no   = (Cp_no   + Cm_no  ).tocsr()

    n_au = poro_s.interface_normals.shape[0]
    print(f"interface nodes: n_u_nodes = {n_au}, n_c (contact) = {n_c_s}")
    assert n_au == n_c_s, f"n_u_nodes != n_c (got {n_au} vs {n_c_s})"
    print(f"C_jmp shapes: with={C_jmp_with.shape}, no={C_jmp_no.shape}")
    print(f"||C_with - C_no||_F                  = {sp.linalg.norm(C_jmp_with - C_jmp_no):.4e}")
    print(f"||C_with - C_no||_F / ||C_no||_F     = "
          f"{sp.linalg.norm(C_jmp_with - C_jmp_no) / sp.linalg.norm(C_jmp_no):.4e}")

    # --- rotation xy -> (n,t), then permute component-major -> interlaced --
    rows_R, cols_R, data_R = [], [], []
    for k in range(n_au):
        nx, ny = poro_s.interface_normals[k]
        rows_R += [k, k]
        cols_R += [0 * n_au + k, 1 * n_au + k]
        data_R += [nx, ny]
        rows_R += [n_au + k, n_au + k]
        cols_R += [0 * n_au + k, 1 * n_au + k]
        data_R += [-ny, nx]
    R_xy_to_nt = sp.csr_matrix(
        (data_R, (rows_R, cols_R)), shape=(2 * n_au, 2 * n_au))
    perm_idx = [r for k in range(n_au) for r in (k, n_au + k)]
    P_cm_to_il = sp.csr_matrix(
        (np.ones(len(perm_idx)), (np.arange(len(perm_idx)), perm_idx)),
        shape=(len(perm_idx), len(perm_idx)),
    )

    D_no   = (P_cm_to_il @ R_xy_to_nt @ C_jmp_no  ).tocsr()
    D_with = (P_cm_to_il @ R_xy_to_nt @ C_jmp_with).tocsr()

    # --- recover u_orig from converged y_sliding[-1] ----------------------
    y_final = np.asarray(y_sliding[-1], dtype=float)
    y_static = y_final[:n_base_s]
    print(f"y_static.shape = {y_static.shape}, "
          f"poro_s._T_inv.shape = {poro_s._T_inv.shape}")
    T_inv_u_block = poro_s._T_inv[Np_s:Np_s + Nu_s, :].tocsr()
    print(f"T_inv_u_block.shape = {T_inv_u_block.shape}")
    u_orig = np.asarray(T_inv_u_block @ y_static, dtype=float).ravel()
    print(f"u_orig.shape = {u_orig.shape}, |u_orig|_max = {np.abs(u_orig).max():.3e}")

    # --- delta diagnostic --------------------------------------------------
    delta_plus, delta_minus = asm_audit._compute_interface_delta()
    delta_plus_norm = np.linalg.norm(delta_plus, axis=1)
    h_est = 1.0 / N_ELEM
    print(f"\n|delta|_plus per interface node:")
    print(f"  min  = {delta_plus_norm.min():.3e}")
    print(f"  max  = {delta_plus_norm.max():.3e}")
    print(f"  mean = {delta_plus_norm.mean():.3e}")
    print(f"  reference h ~ 1/N_ELEM = {h_est:.3e}  "
          f"(|delta| should be O(h/2) for SBM)")

    # --- apply both extractors --------------------------------------------
    slip_no   = np.asarray(D_no   @ u_orig, dtype=float)
    slip_with = np.asarray(D_with @ u_orig, dtype=float)
    slip_diff = slip_with - slip_no

    # interlock vs existing extractor
    slip_existing = np.asarray(sliding_slip_history[-1], dtype=float)
    slip_no_t = slip_no[1::2]
    slip_no_n = slip_no[0::2]
    err_intlk = np.linalg.norm(slip_no_t - slip_existing) / max(np.linalg.norm(slip_existing), 1e-30)
    print(f"\nINTERLOCK ||audit_no_taylor (t) - sliding_slip[-1]|| / ||sliding_slip[-1]|| "
          f"= {err_intlk:.3e}")
    print(f"  (should be ~1e-15 if our composition matches the existing chain)")
    print(f"max |normal-jump| (no-Taylor)   = {np.abs(slip_no_n).max():.3e}")
    print(f"max |Taylor-correction-vector|  = {np.abs(slip_diff).max():.3e}")
    print(f"||Taylor correction||_2 / ||slip||_2 = "
          f"{np.linalg.norm(slip_diff) / max(np.linalg.norm(slip_no), 1e-30):.4e}")

    # per-node Taylor magnitude
    diff_n = slip_diff[0::2]
    diff_t = slip_diff[1::2]
    print(f"\nPer-node Taylor correction (interlaced layout):")
    print(f"  max |delta_n| = {np.abs(diff_n).max():.3e}")
    print(f"  max |delta_t| = {np.abs(diff_t).max():.3e}")

    # --- Pollard rel L2 with/without Taylor -------------------------------
    slip_phys_no   = slip_no_t   * DISP_SCALE
    slip_phys_with = slip_with[1::2] * DISP_SCALE
    slip_phys_no_sorted   = slip_phys_no  [_sort_order]
    slip_phys_with_sorted = slip_phys_with[_sort_order]
    slip_anal = slip_max_anal * np.sqrt(np.clip(1.0 - s_param ** 2, 0.0, None))
    mask = np.abs(s_param) < 0.95
    err_no_   = float(np.linalg.norm(np.abs(slip_phys_no_sorted  [mask]) - np.abs(slip_anal[mask])))
    err_with_ = float(np.linalg.norm(np.abs(slip_phys_with_sorted[mask]) - np.abs(slip_anal[mask])))
    ref       = float(np.linalg.norm(np.abs(slip_anal[mask])))

    print(f"\n=== POLLARD COMPARISON (interior |s| < 0.95) ===")
    print(f"surrogate-frame  (no  Taylor): rel L2 = {err_no_   / ref:.4f}")
    print(f"true-crack-frame (with Taylor): rel L2 = {err_with_ / ref:.4f}")
    print(f"existing notebook reported   : rel L2 = {err_l2 / ref:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
