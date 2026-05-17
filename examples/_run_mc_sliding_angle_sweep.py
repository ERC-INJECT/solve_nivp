"""Angle sweep: confirm SBM bias scales as cos(theta_crack).

For each (USE_SBM, theta_crack_deg, N) triple, inject:
  1. override after the user-inputs cell (cell 5): USE_SBM, INCLUDE_SBM=False,
     INCLUDE_TAYLOR=True, N_ELEM
  2. theta override after cell 7 (where THETA_SLIDING is defined):
     THETA_SLIDING = pi/2 - radians(theta_crack_deg)
     re-derive sN_sliding / tau_sliding / CRACK_THETA_SLIDING
  3. dump cell at end (same JSON sidecar layout as the v2 sweep).

Subprocess-isolated `jupyter nbconvert --execute`, one fresh kernel per case.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
SRC_NB = ROOT / "embedded_crack_mc_sliding.ipynb"
OUT_DIR = ROOT / "_sweep_runs"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    # tag,                        N,  USE_SBM, theta_crack_deg
    ("angle15_conformal_N40",     40, False,   15),
    ("angle15_sbm_N40",           40, True,    15),
    ("angle45_conformal_N40",     40, False,   45),
    ("angle45_sbm_N40",           40, True,    45),
]

KERNEL_TIMEOUT_S = 60 * 60 * 2


def find_cell_index_with(nb, *substrings) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if all(s in c.source for s in substrings):
            return i
    raise RuntimeError(f"could not find cell containing all of {substrings!r}")


def make_param_override_cell(n_elem: int, use_sbm: bool):
    return nbformat.v4.new_code_cell(source=(
        "# --- sweep override (params) ---\n"
        f"USE_SBM = {bool(use_sbm)}\n"
        "CONFORMING_CRACK_MESH = not USE_SBM\n"
        "INCLUDE_SBM = False\n"
        "INCLUDE_TAYLOR = True\n"
        f"N_ELEM = {int(n_elem)}\n"
    ))


def make_theta_override_cell(theta_crack_deg: float):
    return nbformat.v4.new_code_cell(source=(
        "# --- sweep override (theta_crack) ---\n"
        f"_theta_crack_deg = {float(theta_crack_deg)}\n"
        "THETA_SLIDING = np.pi/2 - np.radians(_theta_crack_deg)\n"
        "sN_sliding, tau_sliding = mohr_tractions(THETA_SLIDING)\n"
        "CRACK_THETA_SLIDING = np.pi / 2 - THETA_SLIDING\n"
        "print(f'[override] theta_crack = {_theta_crack_deg} deg, "
        "THETA_SLIDING = {np.degrees(THETA_SLIDING):.1f} deg, '\n"
        "      f'sN={sN_sliding:.3f}, tau={tau_sliding:.3f}')\n"
    ))


def make_dump_cell(json_path: Path):
    return nbformat.v4.new_code_cell(source=(
        "# --- sweep dump ---\n"
        "import json, numpy as np\n"
        "_dump = {\n"
        "    'USE_SBM': bool(USE_SBM),\n"
        "    'CONFORMING_CRACK_MESH': bool(CONFORMING_CRACK_MESH),\n"
        "    'INCLUDE_SBM': bool(INCLUDE_SBM),\n"
        "    'INCLUDE_TAYLOR': bool(INCLUDE_TAYLOR),\n"
        "    'N_ELEM': int(N_ELEM),\n"
        "    'MU': float(MU),\n"
        "    'CRACK_LENGTH': float(CRACK_LENGTH),\n"
        "    'G_SHEAR': float(G_SHEAR),\n"
        "    'NU': float(NU),\n"
        "    'THETA_SLIDING_deg': float(np.degrees(THETA_SLIDING)),\n"
        "    'CRACK_THETA_SLIDING_deg': float(np.degrees(CRACK_THETA_SLIDING)),\n"
        "    'sigma_N_sliding': float(sigma_N_sliding),\n"
        "    'tau_sliding': float(tau_sliding_val),\n"
        "    'slip_max_anal': float(slip_max_anal),\n"
        "    's_param_with_tips': np.asarray(s_param_with_tips).tolist(),\n"
        "    'slip_with_tips_phys': np.asarray(slip_with_tips).tolist(),\n"
        "    'err_l2': float(err_l2),\n"
        "    'ref_l2': float(ref_l2),\n"
        "    'rel_l2': float(err_l2 / max(ref_l2, 1e-30)),\n"
        "}\n"
        f"with open(r'{json_path}', 'w') as f:\n"
        "    json.dump(_dump, f, indent=2)\n"
        f"print('[sweep dump] wrote', r'{json_path}')\n"
    ))


def prepare_notebook(tag, n_elem, use_sbm, theta_crack_deg):
    out_nb = OUT_DIR / f"mc_sliding_{tag}.ipynb"
    out_json = OUT_DIR / f"mc_sliding_{tag}.json"
    log_path = OUT_DIR / f"mc_sliding_{tag}.log"

    nb = nbformat.read(SRC_NB, as_version=4)
    # First inject theta override AFTER the cell that defines THETA_SLIDING.
    theta_idx = find_cell_index_with(nb, "THETA_SLIDING", "np.radians", "theta_onset")
    nb.cells.insert(theta_idx + 1, make_theta_override_cell(theta_crack_deg))

    # Then inject param override AFTER the user-inputs cell.  Recompute index
    # after the previous insertion so we still find the right cell.
    param_idx = find_cell_index_with(nb, "USE_SBM", "N_ELEM", "INCLUDE_SBM")
    nb.cells.insert(param_idx + 1, make_param_override_cell(n_elem, use_sbm))

    nb.cells.append(make_dump_cell(out_json))
    nbformat.write(nb, out_nb)
    return out_nb, out_json, log_path


def run_one(tag, n_elem, use_sbm, theta_crack_deg) -> int:
    out_nb, out_json, log_path = prepare_notebook(tag, n_elem, use_sbm, theta_crack_deg)
    if out_json.exists():
        out_json.unlink()
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={KERNEL_TIMEOUT_S}",
        "--ExecutePreprocessor.iopub_timeout=300",
        str(out_nb),
    ]
    print(f"[{time.strftime('%H:%M:%S')}] starting {tag} (theta={theta_crack_deg} deg)", flush=True)
    t0 = time.time()
    with open(log_path, "wb") as logf:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.time() - t0
    rc = proc.returncode
    print(
        f"[{time.strftime('%H:%M:%S')}] finished {tag}  rc={rc}  "
        f"json={'yes' if out_json.exists() else 'NO'}  elapsed={elapsed:.1f}s",
        flush=True,
    )
    return rc


def main() -> int:
    failures = 0
    for tag, n, sbm, theta in CASES:
        rc = run_one(tag, n, sbm, theta)
        if rc != 0:
            failures += 1
    print(f"[{time.strftime('%H:%M:%S')}] all done  failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
