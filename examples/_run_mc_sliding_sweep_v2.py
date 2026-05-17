"""Robust serial sweep over examples/embedded_crack_mc_sliding.ipynb.

Each case spawns a fresh Python subprocess running `jupyter nbconvert --execute`,
so kernel-driver deadlocks (the stall we hit with in-process nbclient) cannot
persist across cases.  The pre-injected notebook is written to a temp file,
the subprocess executes it in-place, and a dump cell at the end writes a
JSON sidecar with the slip profile and Pollard L2 numbers.

Usage:
    python examples/_run_mc_sliding_sweep_v2.py [tag1 tag2 ...]

If no tags are given, runs all 6 cases. Tags are like "conformal_N20", "sbm_N40".
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

ALL_CASES = [
    ("conformal_N20", 20, False),
    ("conformal_N30", 30, False),
    ("conformal_N40", 40, False),
    ("conformal_N50", 50, False),
    ("conformal_N60", 60, False),
    ("conformal_N80", 80, False),
    ("sbm_N20",       20, True),
    ("sbm_N30",       30, True),
    ("sbm_N40",       40, True),
    ("sbm_N50",       50, True),
    ("sbm_N60",       60, True),
    ("sbm_N80",       80, True),
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


def prepare_notebook(tag: str, n_elem: int, use_sbm: bool):
    out_nb = OUT_DIR / f"mc_sliding_{tag}.ipynb"
    out_json = OUT_DIR / f"mc_sliding_{tag}.json"
    log_path = OUT_DIR / f"mc_sliding_{tag}.log"

    nb = nbformat.read(SRC_NB, as_version=4)
    idx = find_param_cell_index(nb)
    nb.cells.insert(idx + 1, make_override_cell(n_elem, use_sbm))
    nb.cells.append(make_dump_cell(out_json))
    nbformat.write(nb, out_nb)
    return out_nb, out_json, log_path


def run_one(tag: str, n_elem: int, use_sbm: bool) -> int:
    out_nb, out_json, log_path = prepare_notebook(tag, n_elem, use_sbm)
    if out_json.exists():
        out_json.unlink()

    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={KERNEL_TIMEOUT_S}",
        f"--ExecutePreprocessor.iopub_timeout=300",
        str(out_nb),
    ]
    print(f"[{time.strftime('%H:%M:%S')}] starting {tag}", flush=True)
    t0 = time.time()
    with open(log_path, "wb") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    elapsed = time.time() - t0
    rc = proc.returncode
    json_ok = out_json.exists()
    print(
        f"[{time.strftime('%H:%M:%S')}] finished {tag}  rc={rc}  "
        f"json={'yes' if json_ok else 'NO'}  elapsed={elapsed:.1f}s  log={log_path}",
        flush=True,
    )
    return rc


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        wanted = set(argv[1:])
        cases = [c for c in ALL_CASES if c[0] in wanted]
        if not cases:
            print(f"no matching cases for {wanted}; available: {[c[0] for c in ALL_CASES]}")
            return 2
    else:
        cases = ALL_CASES

    print(f"source notebook: {SRC_NB}")
    print(f"output dir:      {OUT_DIR}")
    print(f"cases:           {[c[0] for c in cases]}")
    failures = 0
    for tag, n, sbm in cases:
        rc = run_one(tag, n, sbm)
        if rc != 0:
            failures += 1
    print(f"[{time.strftime('%H:%M:%S')}] all done  failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
