"""Surgical patch: replace inj_post with the bug-fixed version, and insert
a new inj_dashboard cell right after.  Leaves inj_params (and any other
user-modified cells) untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

# Pull the canonical CELL_POST_INJ and CELL_DASHBOARD_INJ from the builder
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_inject_section_cells",
    "/home/david/Documents/Solve_ivp_ns/examples/_inject_section_cells.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CELL_POST_INJ = _mod.CELL_POST_INJ
CELL_DASHBOARD_INJ = _mod.CELL_DASHBOARD_INJ

NB_PATH = Path("/home/david/Documents/Solve_ivp_ns/examples/embedded_crack_mohr_coulomb_ncp.ipynb")

def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Replace existing inj_post in place
    post_idx = next((i for i, c in enumerate(cells) if c.get("id") == "inj_post"), None)
    if post_idx is None:
        raise RuntimeError("inj_post cell not found in notebook")
    cells[post_idx] = CELL_POST_INJ

    # Drop any existing inj_dashboard, then insert fresh one after inj_post
    cells = [c for c in cells if c.get("id") != "inj_dashboard"]
    post_idx = next(i for i, c in enumerate(cells) if c.get("id") == "inj_post")
    cells.insert(post_idx + 1, CELL_DASHBOARD_INJ)

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"replaced inj_post and inserted inj_dashboard; total cells now {len(cells)}")

if __name__ == "__main__":
    main()
