"""Serial N-sweep over the embedded-crack Mohr-Coulomb NCP notebook.

Executes examples/embedded_crack_mohr_coulomb_ncp.ipynb once per (N_ELEM, INCLUDE_SBM)
pair, by inserting an override cell after the user-input cell. Output notebooks land
in examples/_sweep_runs/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
SRC_NB = ROOT / "embedded_crack_mohr_coulomb_ncp.ipynb"
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
        if c.cell_type == "code" and "N_ELEM" in c.source and "INCLUDE_SBM" in c.source:
            return i
    raise RuntimeError("could not locate parameter cell containing N_ELEM and INCLUDE_SBM")


def make_override_cell(n_elem: int, include_sbm: bool):
    src = (
        "# --- sweep override (injected) ---\n"
        f"N_ELEM = {n_elem}\n"
        f"INCLUDE_SBM = {bool(include_sbm)}\n"
    )
    return nbformat.v4.new_code_cell(source=src)


def run_one(n_elem: int, include_sbm: bool) -> None:
    tag = f"{'sbm' if include_sbm else 'conformal'}_N{n_elem}"
    out_path = OUT_DIR / f"embedded_crack_{tag}.ipynb"

    nb = nbformat.read(SRC_NB, as_version=4)
    idx = find_param_cell_index(nb)
    nb.cells.insert(idx + 1, make_override_cell(n_elem, include_sbm))

    client = NotebookClient(
        nb,
        timeout=KERNEL_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] starting {tag}", flush=True)
    t0 = time.time()
    try:
        client.execute()
        status = "OK"
    except CellExecutionError as e:
        status = f"FAILED: {e.ename}"
    finally:
        nbformat.write(nb, out_path)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] finished {tag}  status={status}  "
          f"elapsed={elapsed:.1f}s  out={out_path}", flush=True)


def main() -> int:
    print(f"source notebook: {SRC_NB}")
    print(f"output dir:      {OUT_DIR}")
    print(f"cases:           {CASES}")
    for n_elem, include_sbm in CASES:
        run_one(n_elem, include_sbm)
    print(f"[{time.strftime('%H:%M:%S')}] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
