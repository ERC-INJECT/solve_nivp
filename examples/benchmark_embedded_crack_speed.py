"""Speed benchmark for the embedded-crack Mohr-Coulomb notebook.

The benchmark intentionally reuses the dated notebook setup so that timing
comparisons stay tied to the current validation scenario.  It defaults to
N_ELEM=40, builds the sliding case once, then reruns the contact-system setup
for each solver mode to avoid cross-case warm-start contamination.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NB = ROOT / "examples" / "embedded_crack_mohr_coulomb_ncp_constant_mu_2026-04-24.ipynb"


@dataclass
class BenchmarkRow:
    case: str
    status: str
    n_elem: int
    tmax: float
    h0: float
    t_eval_dt: float | None
    adaptive: bool
    linear_solver: str
    elapsed_s: float | None = None
    build_contact_s: float | None = None
    t_final: float | None = None
    stored_states: int | None = None
    accepted_steps: int | None = None
    h_min: float | None = None
    h_median: float | None = None
    h_max: float | None = None
    adaptive_attempts: int | None = None
    adaptive_accepted: int | None = None
    adaptive_rejected: int | None = None
    attempt_newton_mean: float | None = None
    attempt_newton_max: int | None = None
    newton_mean: float | None = None
    newton_max: int | None = None
    normal_jump_maxabs: float | None = None
    slip_maxabs: float | None = None
    cone_margin_min_mpa: float | None = None
    cone_margin_final_maxabs_mpa: float | None = None
    constraint_inf: float | None = None
    mesh_vertices: int | None = None
    split_vertices: int | None = None
    mesh_elements: int | None = None
    n_base: int | None = None
    n_phys: int | None = None
    n_aug: int | None = None
    n_contact: int | None = None
    solver_profile: dict[str, Any] | None = None
    error: str | None = None


def load_notebook() -> dict[str, Any]:
    return json.loads(NB.read_text())


def run_cell(nb: dict[str, Any], ns: dict[str, Any], idx: int) -> None:
    code = "".join(nb["cells"][idx].get("source", []))
    print(f"\n--- executing notebook cell {idx} ---", flush=True)
    exec(compile(code, f"{NB}:cell-{idx}", "exec"), ns)


def prepare_context(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    nb = load_notebook()
    sys.path.insert(0, str(SRC))
    ns: dict[str, Any] = {"__name__": "__embedded_crack_speed__"}

    run_cell(nb, ns, 2)
    run_cell(nb, ns, 4)
    ns["N_ELEM"] = int(args.n_elem)
    ns["TMAX"] = float(args.tmax)
    ns["H_FIXED"] = float(args.h)
    ns["BULK_MU_V"] = float(args.bulk_mu_v)
    ns["BULK_LAM_V"] = float(args.bulk_lam_v)
    print(
        f"Benchmark overrides: N_ELEM={ns['N_ELEM']}, "
        f"TMAX={ns['TMAX']}, H_FIXED={ns['H_FIXED']}, "
        f"bulk_mu_v={ns['BULK_MU_V']}, bulk_lam_v={ns['BULK_LAM_V']}",
        flush=True,
    )
    run_cell(nb, ns, 6)

    setup_start = time.perf_counter()
    run_cell(nb, ns, 27)
    setup_elapsed = time.perf_counter() - setup_start
    return ns, setup_elapsed


def case_config(name: str) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {
        "splu_fixed": {
            "adaptive": False,
            "linear_solver": "splu",
            "petsc_options": None,
        },
        "splu_adaptive": {
            "adaptive": True,
            "linear_solver": "splu",
            "petsc_options": None,
        },
        "petsc_mumps_fixed": {
            "adaptive": False,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "petsc_mumps_adaptive": {
            "adaptive": True,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "petsc_gmres_ilu_fixed": {
            "adaptive": False,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": 1.0e-6,
                "ksp_max_it": 1000,
                "ksp_gmres_restart": 100,
            },
        },
        "petsc_gmres_jacobi_fixed": {
            "adaptive": False,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "jacobi",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 500,
                "ksp_gmres_restart": 100,
            },
        },
        "petsc_gmres_jacobi_adaptive": {
            "adaptive": True,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "jacobi",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 500,
                "ksp_gmres_restart": 100,
            },
        },
        "petsc_fieldsplit_additive_fixed": {
            "adaptive": False,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 500,
                "ksp_gmres_restart": 100,
                "fieldsplit_ksp_type": "preonly",
                "fieldsplit_pc_type": "jacobi",
            },
        },
        "petsc_fieldsplit_additive_adaptive": {
            "adaptive": True,
            "linear_solver": "petsc",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 500,
                "ksp_gmres_restart": 100,
                "fieldsplit_ksp_type": "preonly",
                "fieldsplit_pc_type": "jacobi",
            },
        },
        "petsc_fieldsplit_phys_contact_schur_fixed": {
            "adaptive": False,
            "linear_solver": "petsc",
            "petsc_field_slices": "phys_contact",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_factorization_type": "full",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 200,
                "ksp_gmres_restart": 100,
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "ilu",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "jacobi",
            },
        },
        "petsc_fieldsplit_phys_contact_schur_adaptive": {
            "adaptive": True,
            "linear_solver": "petsc",
            "petsc_field_slices": "phys_contact",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_factorization_type": "full",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 200,
                "ksp_gmres_restart": 100,
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "ilu",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "jacobi",
            },
        },
        "petsc_fieldsplit_phys_contact_additive_fixed": {
            "adaptive": False,
            "linear_solver": "petsc",
            "petsc_field_slices": "phys_contact",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 500,
                "ksp_gmres_restart": 100,
                "fieldsplit_ksp_type": "preonly",
                "fieldsplit_pc_type": "jacobi",
            },
        },
        "petsc_fieldsplit_phys_contact_additive_adaptive": {
            "adaptive": True,
            "linear_solver": "petsc",
            "petsc_field_slices": "phys_contact",
            "petsc_options": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",
                "ksp_rtol": 1.0e-5,
                "ksp_max_it": 500,
                "ksp_gmres_restart": 100,
                "fieldsplit_ksp_type": "preonly",
                "fieldsplit_pc_type": "jacobi",
            },
        },
    }
    if name not in cases:
        raise ValueError(f"Unknown case {name!r}; choices: {', '.join(sorted(cases))}")
    return cases[name]


def _iter_stats(info: Any, np: Any) -> tuple[float | None, int | None]:
    if not isinstance(info, (list, tuple)) or not info:
        return None, None
    values = []
    for entry in info:
        try:
            values.append(float(entry[2]))
        except Exception:
            pass
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), int(arr.max())


def build_t_eval(np: Any, tmax: float, dt: float) -> Any | None:
    if dt <= 0.0:
        return None

    tol = 1.0e-12 * max(1.0, abs(tmax))
    values = np.arange(0.0, tmax + 0.5 * dt, dt, dtype=float)
    values = values[values <= tmax + tol]
    if values.size == 0:
        values = np.array([0.0, tmax], dtype=float)
    elif abs(values[-1] - tmax) > tol:
        values = np.append(values, tmax)
    else:
        values[-1] = tmax
    return values


def summarize_solution(
    ns: dict[str, Any],
    row: BenchmarkRow,
    t_hist: Any,
    y_hist: Any,
    h_hist: Any,
    info: Any,
    attempts: dict[str, Any] | None,
) -> None:
    np = ns["np"]
    MU = ns["MU"]
    sigma_scale = ns["Sigma_scale"]
    n_phys = int(ns["n_phys_s"])
    n_contact = int(ns["n_c_s"])
    y_arr = np.asarray(y_hist, dtype=float)

    gap = np.asarray(y_arr[:, ns["jmpu_n_idx_s"]], dtype=float)
    slip = np.asarray(y_arr[:, ns["jmpu_t_idx_s"]], dtype=float)
    r_pert, r_total, r_n_nd, r_t_nd = ns["_contact_history"](
        y_arr, n_phys, n_contact, ns["_s0_val_s"], ns["_w0_val_s"]
    )
    r_n = r_n_nd * sigma_scale
    r_t = r_t_nd * sigma_scale
    cone_margin = MU * r_n - np.abs(r_t)
    constraint_inf = ns["_constraint_residual_inf"](
        ns["cs_s"].projected_radau_contact, y_arr[-1, :n_phys]
    )

    h_arr = np.asarray(h_hist, dtype=float) if h_hist is not None else np.array([])
    row.t_final = float(t_hist[-1])
    row.stored_states = int(len(t_hist))
    row.accepted_steps = int(max(0, len(t_hist) - 1))
    if h_arr.size:
        row.h_min = float(h_arr.min())
        row.h_median = float(np.median(h_arr))
        row.h_max = float(h_arr.max())
    row.newton_mean, row.newton_max = _iter_stats(info, np)
    if attempts:
        accepted = np.asarray(attempts.get("accepted", []), dtype=bool)
        row.adaptive_attempts = int(accepted.size)
        row.adaptive_accepted = int(np.count_nonzero(accepted))
        row.adaptive_rejected = int(accepted.size - row.adaptive_accepted)
        row.accepted_steps = row.adaptive_accepted
        if "iterations" in attempts:
            iters = np.asarray(attempts.get("iterations", []), dtype=float)
            valid = iters >= 0
            if np.any(valid):
                row.attempt_newton_mean = float(np.mean(iters[valid]))
                row.attempt_newton_max = int(np.max(iters[valid]))
    row.normal_jump_maxabs = float(np.max(np.abs(gap)))
    row.slip_maxabs = float(np.max(np.abs(slip)))
    row.cone_margin_min_mpa = float(cone_margin.min())
    row.cone_margin_final_maxabs_mpa = float(np.max(np.abs(cone_margin[-1])))
    row.constraint_inf = float(constraint_inf)


def fill_problem_sizes(ns: dict[str, Any], row: BenchmarkRow) -> None:
    row.mesh_vertices = int(ns["mesh_s"].p.shape[1])
    row.split_vertices = int(ns["poro_s"].mesh.p.shape[1])
    row.mesh_elements = int(ns["mesh_s"].t.shape[1])
    row.n_base = int(ns["n_base_s"])
    row.n_phys = int(ns["n_phys_s"])
    row.n_aug = int(ns["cs_s"].y0.size)
    row.n_contact = int(ns["n_c_s"])


def run_case(nb: dict[str, Any], ns: dict[str, Any], args: argparse.Namespace, name: str) -> BenchmarkRow:
    cfg = case_config(name)
    row = BenchmarkRow(
        case=name,
        status="error",
        n_elem=int(args.n_elem),
        tmax=float(args.tmax),
        h0=float(args.h),
        t_eval_dt=float(args.t_eval_dt) if float(args.t_eval_dt) > 0.0 else None,
        adaptive=bool(cfg["adaptive"]),
        linear_solver=str(cfg["linear_solver"]),
    )

    try:
        build_start = time.perf_counter()
        run_cell(nb, ns, 28)
        row.build_contact_s = time.perf_counter() - build_start
        fill_problem_sizes(ns, row)

        np = ns["np"]
        solve_nivp = ns["solve_nivp"]
        cs_s = ns["cs_s"]
        n_aug = cs_s.y0.size
        n_phys = ns["A_dyn_s"].shape[0]

        nl_atol = np.full(n_aug, 1.0e-8)
        nl_rtol = np.full(n_aug, 1.0e-6)
        nl_atol[n_phys:] = 1.0e-10
        nl_rtol[n_phys:] = 0.0

        solver_opts = dict(cs_s.solver_opts)
        solver_opts.pop("cold_start_slices", None)
        solver_opts["damped_step_fraction"] = 1.0
        solver_opts["diagonal_regularization"] = 0.0
        solver_opts["profile"] = bool(args.profile_solver)
        solver_opts.update(
            tol=float(args.solver_tol),
            max_iter=int(args.solver_max_iter),
            rhs_jac=cs_s.rhs_jac,
            linear_solver=str(cfg["linear_solver"]),
        )
        if cfg.get("linear_solver") == "petsc":
            solver_opts["petsc_reuse_steps"] = int(args.petsc_reuse_steps)
            if cfg.get("petsc_options"):
                solver_opts["petsc_options"] = dict(cfg["petsc_options"])
            if cfg.get("petsc_field_slices") == "phys_contact":
                solver_opts["petsc_field_slices"] = [
                    slice(0, int(n_phys)),
                    slice(int(n_phys), int(n_aug)),
                ]

        integrator_opts = dict(cs_s.integrator_opts)
        integrator_opts.update({"stages": 2, "use_coupled_newton": True})
        t_eval = build_t_eval(np, float(args.tmax), float(args.t_eval_dt))

        solve_kwargs = dict(
            fun=cs_s.rhs,
            t_span=(0.0, float(args.tmax)),
            y0=cs_s.y0.copy(),
            method="radau_iia",
            integrator_opts=integrator_opts,
            projection=cs_s.projection,
            solver="semismooth_newton",
            projection_opts={"component_slices": cs_s.component_slices},
            solver_opts=solver_opts,
            h0=float(args.h),
            nl_atol=nl_atol,
            nl_rtol=nl_rtol,
            component_slices=cs_s.component_slices,
            verbose=bool(args.solve_verbose),
            A=cs_s.A,
            dae_var_weight="auto",
            thin_output=int(args.thin_output),
            store_fk=False,
            t_eval=t_eval,
            return_profile=bool(args.profile_solver),
        )
        if cfg["adaptive"]:
            solve_kwargs.update(
                adaptive=True,
                rtol=float(args.adaptive_rtol),
                atol=float(args.adaptive_atol),
                skip_error_indices=[0, len(cs_s.component_slices) - 1],
                active_set_filter=True,
                return_attempts=True,
                adaptive_opts=dict(
                    h0=float(args.h),
                    h_min=float(args.adaptive_h_min),
                    h_max=float(args.adaptive_h_max),
                    h_up=2.0,
                    h_down=0.5,
                    method_order=1,
                ),
            )
        else:
            solve_kwargs.update(adaptive=False, active_set_filter=False)

        print(f"\n=== running {name} ===", flush=True)
        start = time.perf_counter()
        result = solve_nivp.solve_ivp_ns(**solve_kwargs)
        row.elapsed_s = time.perf_counter() - start
        profile = None
        if cfg["adaptive"]:
            if args.profile_solver:
                t_hist, y_hist, h_hist, _fk, info, attempts, profile = result
            else:
                t_hist, y_hist, h_hist, _fk, info, attempts = result
        else:
            if args.profile_solver:
                t_hist, y_hist, h_hist, _fk, info, profile = result
            else:
                t_hist, y_hist, h_hist, _fk, info = result
            attempts = None
        row.solver_profile = profile
        summarize_solution(ns, row, t_hist, y_hist, h_hist, info, attempts)
        target_t = float(args.tmax)
        reached_t = row.t_final if row.t_final is not None else float("-inf")
        if reached_t < target_t - max(1.0e-10, 1.0e-10 * abs(target_t)):
            row.status = "incomplete"
            row.error = f"stopped at t={reached_t:.6g} before requested tmax={target_t:.6g}"
        else:
            row.status = "ok"
    except BaseException as exc:  # benchmark rows should survive failed cases
        row.elapsed_s = row.elapsed_s if row.elapsed_s is not None else None
        row.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"Case {name} failed: {row.error}", flush=True)
    return row


def write_results(rows: list[BenchmarkRow], args: argparse.Namespace, setup_elapsed: float) -> None:
    out_stem = pathlib.Path(args.output_stem)
    if not out_stem.is_absolute():
        out_stem = ROOT / out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "notebook": str(NB),
            "setup_elapsed_s": setup_elapsed,
            "n_elem": int(args.n_elem),
            "tmax": float(args.tmax),
            "h": float(args.h),
            "t_eval_dt": float(args.t_eval_dt) if float(args.t_eval_dt) > 0.0 else None,
            "thin_output": int(args.thin_output),
        },
        "rows": [asdict(row) for row in rows],
    }
    json_path = out_stem.with_suffix(".json")
    csv_path = out_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    fields = list(asdict(rows[0]).keys()) if rows else []
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


def print_summary(rows: list[BenchmarkRow], setup_elapsed: float) -> None:
    print("\n=== benchmark summary ===")
    print(f"setup mesh/system elapsed_s: {setup_elapsed:.3f}")
    for row in rows:
        print(
            f"{row.case:24s} status={row.status:5s} "
            f"elapsed={row.elapsed_s if row.elapsed_s is not None else float('nan'):.3f}s "
            f"steps={row.accepted_steps} "
            f"stored={row.stored_states} "
            f"h=[{row.h_min}, {row.h_median}, {row.h_max}] "
            f"normal_jump={row.normal_jump_maxabs} "
            f"newton_mean/max={row.newton_mean}/{row.newton_max} "
            f"attempt_newton_mean/max={row.attempt_newton_mean}/{row.attempt_newton_max}"
        )
        if row.solver_profile:
            timed = [
                (name, float(val.get("time_s", 0.0)), int(val.get("count", 0)))
                for name, val in row.solver_profile.items()
                if float(val.get("time_s", 0.0)) > 0.0
            ]
            timed.sort(key=lambda item: item[1], reverse=True)
            top = ", ".join(
                f"{name}={seconds:.3f}s/{count}"
                for name, seconds, count in timed[:6]
            )
            print(f"  profile top: {top}")
        if row.error:
            print(f"  error: {row.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-elem", type=int, default=40, help="Baseline mesh size.")
    parser.add_argument("--tmax", type=float, default=60.0, help="Final time for benchmark runs.")
    parser.add_argument("--h", type=float, default=0.1, help="Initial/fixed step size.")
    parser.add_argument("--bulk-mu-v", type=float, default=0.0)
    parser.add_argument("--bulk-lam-v", type=float, default=0.0)
    parser.add_argument("--solver-tol", type=float, default=1.0e-8)
    parser.add_argument("--solver-max-iter", type=int, default=500)
    parser.add_argument("--adaptive-rtol", type=float, default=5.0e-2)
    parser.add_argument("--adaptive-atol", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-h-min", type=float, default=1.0e-5)
    parser.add_argument("--adaptive-h-max", type=float, default=0.8)
    parser.add_argument("--petsc-reuse-steps", type=int, default=20)
    parser.add_argument(
        "--t-eval-dt",
        type=float,
        default=1.0,
        help="Requested output/landing interval; use <= 0 to disable t_eval.",
    )
    parser.add_argument(
        "--thin-output",
        type=int,
        default=1,
        help="Store every Nth accepted step in addition to forced t_eval points.",
    )
    parser.add_argument(
        "--profile-solver",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect nonlinear/PETSc phase timing in benchmark JSON.",
    )
    parser.add_argument("--solve-verbose", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["petsc_mumps_adaptive", "petsc_mumps_fixed"],
        help=(
            "Cases to run. Available: splu_fixed, splu_adaptive, "
            "petsc_mumps_fixed, petsc_mumps_adaptive, petsc_gmres_ilu_fixed, "
            "petsc_gmres_jacobi_fixed, petsc_gmres_jacobi_adaptive, "
            "petsc_fieldsplit_additive_fixed, petsc_fieldsplit_additive_adaptive, "
            "petsc_fieldsplit_phys_contact_schur_fixed, "
            "petsc_fieldsplit_phys_contact_schur_adaptive, "
            "petsc_fieldsplit_phys_contact_additive_fixed, "
            "petsc_fieldsplit_phys_contact_additive_adaptive."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default="examples/embedded_crack_speed_mesh40_tmax60_baseline",
        help="Output path without extension for JSON/CSV results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nb = load_notebook()
    ns, setup_elapsed = prepare_context(args)
    rows = [run_case(nb, ns, args, name) for name in args.cases]
    print_summary(rows, setup_elapsed)
    write_results(rows, args, setup_elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
