"""Serial sweep over examples/embedded_crack_mc_sliding.ipynb.

Runs the sliding test against the Pollard-Segall analytical for
N_ELEM in {20, 30, 40} x USE_SBM in {False, True}, six runs total.

For each run we inject:
  1. an override cell after cell 5 setting USE_SBM, CONFORMING_CRACK_MESH,
     INCLUDE_SBM=False, INCLUDE_TAYLOR=True, N_ELEM;
  2. a dump cell at the very end that writes the slip profile, analytical
     scale, and Pollard L2 numbers to a JSON sidecar.

Outputs:
  examples/_sweep_runs/mc_sliding_{conformal|sbm}_N{N}.ipynb
  examples/_sweep_runs/mc_sliding_{conformal|sbm}_N{N}.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
SRC_NB = ROOT / "embedded_crack_mc_sliding.ipynb"
OUT_DIR = ROOT / "_sweep_runs"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    (20, False),
    (30, False),
    (40, False),
    (20, True),
    (30, True),
    (40, True),
]

KERNEL_TIMEOUT_S = 60 * 60 * 2


def find_param_cell_index(nb) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and "USE_SBM" in c.source and "N_ELEM" in c.source:
            return i
    raise RuntimeError("could not locate parameter cell containing USE_SBM and N_ELEM")


def make_override_cell(n_elem: int, use_sbm: bool):
    src = (
        "# --- sweep override (injected) ---\n"
        f"USE_SBM = {bool(use_sbm)}\n"
        "CONFORMING_CRACK_MESH = not USE_SBM\n"
        "INCLUDE_SBM = False\n"
        "INCLUDE_TAYLOR = True\n"
        f"N_ELEM = {int(n_elem)}\n"
    )
    return nbformat.v4.new_code_cell(source=src)


def make_dump_cell(json_path: Path):
    src = (
        "# --- sweep dump (injected) ---\n"
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
    )
    return nbformat.v4.new_code_cell(source=src)


def run_one(n_elem: int, use_sbm: bool) -> None:
    tag = f"{'sbm' if use_sbm else 'conformal'}_N{n_elem}"
    out_nb = OUT_DIR / f"mc_sliding_{tag}.ipynb"
    out_json = OUT_DIR / f"mc_sliding_{tag}.json"

    nb = nbformat.read(SRC_NB, as_version=4)
    idx = find_param_cell_index(nb)
    nb.cells.insert(idx + 1, make_override_cell(n_elem, use_sbm))
    nb.cells.append(make_dump_cell(out_json))

    client = NotebookClient(
        nb,
        timeout=KERNEL_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] starting {tag}", flush=True)
    t0 = time.time()
    status = "OK"
    try:
        client.execute()
    except CellExecutionError as e:
        status = f"FAILED: {e.ename}"
    finally:
        nbformat.write(nb, out_nb)
    elapsed = time.time() - t0
    print(
        f"[{time.strftime('%H:%M:%S')}] finished {tag}  status={status}  "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def main() -> int:
    print(f"source notebook: {SRC_NB}")
    print(f"output dir:      {OUT_DIR}")
    print(f"cases:           {CASES}")
    for n_elem, use_sbm in CASES:
        run_one(n_elem, use_sbm)
    print(f"[{time.strftime('%H:%M:%S')}] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
