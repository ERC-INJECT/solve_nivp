#!/usr/bin/env python
"""Compare the sliding-block SOC and Alart-Curnier backends fairly.

Runs both one-step patch-test helpers with the same:

- mesh type
- mesh resolution
- initial gap
- top/bottom boundary conditions
- body force
- friction coefficient
- fixed backward-Euler step

The goal is to separate "different contact law" from "different demo /
solver settings" as much as possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PORO_ROOT = Path("/home/david/Documents/Poroelasticity")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(REPO_ROOT))
if PORO_ROOT.exists():
    sys.path.insert(0, str(PORO_ROOT))

from sliding_block_one_step_patch_test import run_one_step_case as run_soc_case
from sliding_block_one_step_patch_test_alart_curnier import (
    run_one_step_case as run_ac_case,
)


@dataclass
class BackendSummary:
    backend: str
    success: bool
    iterations: int
    solver_error: float | None
    true_residual_inf: float | None
    true_residual_rms: float | None
    final_gap_min: float
    final_gap_max: float
    max_penetration: float
    max_open_reaction: float
    min_pn: float
    max_abs_pt: float
    wall_seconds: float


def _aligned_soc_solver_overrides() -> dict:
    return {
        "max_iter": 30,
        "adaptive_lam": False,
        "lam_update_strategy": "none",
        "globalization": "linesearch",
        "use_broyden": False,
        "linear_solver": "splu",
        "sparse": True,
        "precond_reuse_steps": 50,
    }


def _aligned_ac_solver_overrides() -> dict:
    return {
        "max_iter": 30,
        "globalization": "linesearch",
        "use_broyden": False,
        "linear_solver": "splu",
        "sparse": True,
        "precond_reuse_steps": 50,
    }


def run_aligned_one_step_comparison(
    *,
    mu_friction: float,
    initial_gap_phys: float,
    t_end_hours: float,
    reverse_gap_sign: bool,
    top_v2_rate: float,
    rho_g: float,
    n_elem: int,
    element_type: str,
) -> tuple[BackendSummary, BackendSummary]:
    common = dict(
        mu_friction=mu_friction,
        initial_gap_phys=initial_gap_phys,
        t_end_hours=t_end_hours,
        reverse_gap_sign=reverse_gap_sign,
        top_v2_rate=top_v2_rate,
        rho_g=rho_g,
        n_elem=n_elem,
        element_type=element_type,
    )

    def _run(label: str, fn, overrides: dict) -> BackendSummary:
        t0 = time.perf_counter()
        res = fn(label=label, solver_overrides=overrides, **common)
        wall = time.perf_counter() - t0
        return BackendSummary(
            backend=label,
            success=bool(res.success),
            iterations=int(res.iterations),
            solver_error=(float(res.solver_error) if res.solver_error is not None else None),
            true_residual_inf=(float(res.true_residual_inf) if res.true_residual_inf is not None else None),
            true_residual_rms=(float(res.true_residual_rms) if res.true_residual_rms is not None else None),
            final_gap_min=float(res.final_gap_min),
            final_gap_max=float(res.final_gap_max),
            max_penetration=float(res.max_penetration),
            max_open_reaction=float(res.max_open_reaction),
            min_pn=float(res.min_pn),
            max_abs_pt=float(res.max_abs_pt),
            wall_seconds=float(wall),
        )

    soc = _run("soc", run_soc_case, _aligned_soc_solver_overrides())
    ac = _run("alart_curnier", run_ac_case, _aligned_ac_solver_overrides())
    return soc, ac


def _print_summary(title: str, items: list[BackendSummary]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for item in items:
        print(f"\n[{item.backend}]")
        print(f"  success:             {item.success}")
        print(f"  iterations:          {item.iterations}")
        print(f"  solver_error:        {item.solver_error}")
        print(f"  true residual inf:   {item.true_residual_inf:.4e}")
        print(f"  true residual rms:   {item.true_residual_rms:.4e}")
        print(
            f"  final gap range:     "
            f"[{item.final_gap_min:+.4e}, {item.final_gap_max:+.4e}]"
        )
        print(f"  max penetration:     {item.max_penetration:.4e}")
        print(f"  max ||p|| on open:   {item.max_open_reaction:.4e}")
        print(f"  min p_N:             {item.min_pn:+.4e}")
        print(f"  max |p_T|:           {item.max_abs_pt:.4e}")
        print(f"  wall time [s]:       {item.wall_seconds:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=0.0, help="Friction coefficient.")
    parser.add_argument("--initial-gap", type=float, default=0.0, help="Initial physical gap [km].")
    parser.add_argument("--t-end-hours", type=float, default=0.01, help="One-step horizon [hours].")
    parser.add_argument("--top-v2-rate", type=float, default=-5.0e-6, help="Top vertical rate [km/hr].")
    parser.add_argument("--rho-g", type=float, default=0.0, help="Body-force magnitude [MPa/km].")
    parser.add_argument("--n-elem", type=int, default=12, help="Elements per side.")
    parser.add_argument(
        "--element-type",
        choices=("tri", "quad"),
        default="tri",
        help="Bulk mesh element type shared by both backends.",
    )
    parser.add_argument(
        "--reverse-gap-sign",
        action="store_true",
        help="Use raw [[u_n]] instead of gap=-[[u_n]].",
    )
    args = parser.parse_args()

    print("Aligned one-step backend comparison")
    print(f"  mu = {args.mu}")
    print(f"  initial_gap = {args.initial_gap}")
    print(f"  t_end_hours = {args.t_end_hours}")
    print(f"  top_v2_rate = {args.top_v2_rate}")
    print(f"  rho_g = {args.rho_g}")
    print(f"  n_elem = {args.n_elem}")
    print(f"  element_type = {args.element_type}")
    print(f"  reverse_gap_sign = {args.reverse_gap_sign}")
    print("  solver alignment:")
    print("    SOC: adaptive_lam=False, globalization=linesearch, sparse=True, SPLU")
    print("    AC : globalization=linesearch, sparse=True, SPLU")

    soc, ac = run_aligned_one_step_comparison(
        mu_friction=args.mu,
        initial_gap_phys=args.initial_gap,
        t_end_hours=args.t_end_hours,
        reverse_gap_sign=bool(args.reverse_gap_sign),
        top_v2_rate=args.top_v2_rate,
        rho_g=args.rho_g,
        n_elem=args.n_elem,
        element_type=args.element_type,
    )
    _print_summary("Aligned Results", [soc, ac])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
