#!/usr/bin/env python
"""Standalone bouncing-ball backend validation script.

Compares the three impact-capable backends used in the bouncing-ball helper:

- `backward_euler` + impulse contact
- `sdirk2` + impulse contact
- `rattle`

It runs:

1. A persistent-contact case
2. A free-fall bounce case with restitution

and prints both the per-case summary table and an optional convergence sweep.
Plots are saved to `examples/bouncing_ball_backend_validation_*.png` by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from bouncing_ball_backend_helpers import (  # noqa: E402
    convergence_sweep_dataframe,
    make_free_fall_case,
    make_persistent_contact_case,
    run_case_bundle,
    results_summary_dataframe,
)


def _print_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:.6g}",
    ):
        print(df.to_string(index=False))


def _plot_case(results: dict[str, dict[str, object]], case_name: str, out_dir: Path) -> Path:
    first = next(iter(results.values()))
    t_ref = np.asarray(first["times"], dtype=float)
    q_ref = np.asarray(first["q_y_ref"], dtype=float)
    v_ref = np.asarray(first["v_y_ref"], dtype=float)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    ax[0, 0].plot(t_ref, q_ref, "k--", linewidth=1.5, label="reference")
    for result in results.values():
        t = np.asarray(result["times"], dtype=float)
        q = np.asarray(result["q_y"], dtype=float)
        ax[0, 0].plot(t, q, label=result["backend_label"])
    ax[0, 0].set_ylabel("height q_y")
    ax[0, 0].set_title("Vertical Position")
    ax[0, 0].legend()

    ax[0, 1].plot(t_ref, v_ref, "k--", linewidth=1.5, label="reference")
    for result in results.values():
        t = np.asarray(result["times"], dtype=float)
        v = np.asarray(result["v_y"], dtype=float)
        ax[0, 1].plot(t, v, label=result["backend_label"])
    ax[0, 1].set_ylabel("velocity v_y")
    ax[0, 1].set_title("Vertical Velocity")
    ax[0, 1].legend()

    for result in results.values():
        t = np.asarray(result["times"], dtype=float)
        q = np.asarray(result["q_y"], dtype=float)
        q_exact = np.asarray(result["q_y_ref"], dtype=float)
        ax[1, 0].plot(t, q - q_exact, label=result["backend_label"])
    ax[1, 0].set_xlabel("time")
    ax[1, 0].set_ylabel("q_y - q_ref")
    ax[1, 0].set_title("Position Error")
    ax[1, 0].legend()

    for result in results.values():
        t = np.asarray(result["times"], dtype=float)
        r = np.asarray(result["r_n"], dtype=float)
        ax[1, 1].plot(t, r, label=result["backend_label"])
    ax[1, 1].set_xlabel("time")
    ax[1, 1].set_ylabel("normal reaction / impulse density proxy")
    ax[1, 1].set_title("Normal Contact Signal")
    ax[1, 1].legend()

    fig.suptitle(f"Bouncing Ball Backend Validation: {case_name}")
    fig.tight_layout()

    out_path = out_dir / f"bouncing_ball_backend_validation_{case_name}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_convergence(df: pd.DataFrame, case_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    metrics = [
        ("max_abs_q_error", "max |q error|"),
        ("impact_time_error", "impact time error"),
        ("wall_time_s", "wall time [s]"),
    ]
    for axis, (metric, title) in zip(ax, metrics, strict=True):
        for backend_label, group in df.groupby("backend_label", sort=False):
            g = group.sort_values("dt", ascending=False)
            axis.loglog(g["dt"], g[metric], marker="o", label=backend_label)
        axis.set_xlabel("dt")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    fig.suptitle(f"Bouncing Ball Convergence: {case_name}")
    fig.tight_layout()

    out_path = out_dir / f"bouncing_ball_backend_convergence_{case_name}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _run_case(
    *,
    case_name: str,
    case: dict[str, object],
    solvers: tuple[str, ...],
    n_steps: int,
    solver_max_iter: int,
    sweep_steps: tuple[int, ...],
    out_dir: Path,
    save_plots: bool,
) -> None:
    results = run_case_bundle(
        case,
        solvers=solvers,
        n_steps=n_steps,
        solver_max_iter=solver_max_iter,
    )
    summary_df = results_summary_dataframe(results)
    summary_cols = [
        "backend_label",
        "success",
        "mean_step_iterations",
        "wall_time_s",
        "max_abs_q_error",
        "max_abs_v_error",
        "impact_time_error",
        "rebound_apex_error",
        "max_penetration",
        "final_qy",
        "final_vy",
        "final_rn",
    ]
    keep_cols = [col for col in summary_cols if col in summary_df.columns]
    _print_df(f"{case_name} Summary", summary_df[keep_cols])

    sweep_df = convergence_sweep_dataframe(
        case,
        solvers=solvers,
        step_counts=sweep_steps,
        solver_max_iter=solver_max_iter,
    )
    sweep_cols = [
        "backend_label",
        "n_steps",
        "dt",
        "success",
        "mean_step_iterations",
        "wall_time_s",
        "max_abs_q_error",
        "max_abs_v_error",
        "impact_time_error",
        "rebound_apex_error",
        "max_penetration",
    ]
    keep_sweep_cols = [col for col in sweep_cols if col in sweep_df.columns]
    _print_df(f"{case_name} Convergence Sweep", sweep_df[keep_sweep_cols])

    if save_plots:
        traj_path = _plot_case(results, case_name, out_dir)
        conv_path = _plot_convergence(sweep_df, case_name, out_dir)
        print(f"\nsaved plots:")
        print(f"  {traj_path}")
        print(f"  {conv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("persistent", "free-fall", "both"),
        default="both",
        help="Which bouncing-ball scenario to run.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("backward_euler", "sdirk2", "rattle"),
        default=("backward_euler", "sdirk2", "rattle"),
        help="Backends to compare.",
    )
    parser.add_argument("--n-steps", type=int, default=1000, help="Main comparison step count.")
    parser.add_argument(
        "--sweep-steps",
        type=int,
        nargs="+",
        default=(250, 500, 1000, 2000),
        help="Step counts used for the convergence sweep.",
    )
    parser.add_argument("--solver-max-iter", type=int, default=200, help="Max nonlinear iterations.")
    parser.add_argument("--t-end", type=float, default=1.0, help="Final time for both cases.")
    parser.add_argument("--height0", type=float, default=1.0, help="Initial height for free-fall.")
    parser.add_argument("--vy0", type=float, default=0.0, help="Initial vertical velocity for free-fall.")
    parser.add_argument("--restitution", type=float, default=0.8, help="Restitution coefficient for free-fall.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving trajectory and convergence plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLES_DIR,
        help="Directory for saved plot files.",
    )
    args = parser.parse_args()

    solvers = tuple(args.solvers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Bouncing-ball backend validation")
    print(f"  case = {args.case}")
    print(f"  solvers = {solvers}")
    print(f"  n_steps = {args.n_steps}")
    print(f"  sweep_steps = {tuple(args.sweep_steps)}")
    print(f"  solver_max_iter = {args.solver_max_iter}")
    print(f"  t_end = {args.t_end}")
    if args.case in {"free-fall", "both"}:
        print(f"  height0 = {args.height0}")
        print(f"  vy0 = {args.vy0}")
        print(f"  restitution = {args.restitution}")

    if args.case in {"persistent", "both"}:
        persistent_case = make_persistent_contact_case(t_end=args.t_end)
        _run_case(
            case_name="persistent_contact",
            case=persistent_case,
            solvers=solvers,
            n_steps=args.n_steps,
            solver_max_iter=args.solver_max_iter,
            sweep_steps=tuple(args.sweep_steps),
            out_dir=out_dir,
            save_plots=not args.no_plots,
        )

    if args.case in {"free-fall", "both"}:
        free_fall_case = make_free_fall_case(
            t_end=args.t_end,
            height0=args.height0,
            vy0=args.vy0,
            restitution=args.restitution,
        )
        _run_case(
            case_name="free_fall_bounce",
            case=free_fall_case,
            solvers=solvers,
            n_steps=args.n_steps,
            solver_max_iter=args.solver_max_iter,
            sweep_steps=tuple(args.sweep_steps),
            out_dir=out_dir,
            save_plots=not args.no_plots,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
