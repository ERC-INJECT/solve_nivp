"""Sequential Mohr-Coulomb convergence test driver.

This runner is intended for HPC sweeps of the sliding prestress example in
``convergence_mohr_coulomb_alpha0.py``.  It runs each requested case and mesh
level sequentially, saves slip profiles against the analytical solution, and
writes explicit parametric-coordinate L2 errors.

Example:

    python examples/run_mohr_coulomb_convergence_test.py \
        --levels 20,30,40,50,60,70,80 \
        --cases conforming_tri,conforming_quad,structured_tri,structured_quad \
        --cores 32 \
        --out examples/_audit_mohr_coulomb_convergence_test

Use scheduler CPU allocation such as ``srun -c 32`` or equivalent.  The current
solver path uses PETSc/MUMPS through PETSc COMM_SELF, so do not use
``mpiexec -n 32`` for this script unless the solver is updated to support
distributed PETSc communicators.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


MPI_RANK_ENV_KEYS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)


@dataclass(frozen=True)
class CaseConfig:
    name: str
    label: str
    element_type: str
    mesh_source: str
    use_sbm: bool


CASE_CONFIGS = {
    "conforming_tri": CaseConfig(
        name="conforming_tri",
        label="conforming tri",
        element_type="tri",
        mesh_source="auto",
        use_sbm=False,
    ),
    "conforming_quad": CaseConfig(
        name="conforming_quad",
        label="conforming quad",
        element_type="quad",
        mesh_source="auto",
        use_sbm=False,
    ),
    "structured_tri": CaseConfig(
        name="structured_tri",
        label="structured shifted tri",
        element_type="tri",
        mesh_source="scikit",
        use_sbm=True,
    ),
    "structured_quad": CaseConfig(
        name="structured_quad",
        label="structured shifted quad",
        element_type="quad",
        mesh_source="scikit",
        use_sbm=True,
    ),
}


def _split_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _split_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _mpi_rank_from_env() -> int:
    for key in MPI_RANK_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _configure_thread_env(cores: int) -> None:
    cores_s = str(int(cores))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = cores_s


def _load_convergence_module(repo_root: Path):
    module_path = repo_root / "examples" / "convergence_mohr_coulomb_alpha0.py"
    spec = importlib.util.spec_from_file_location("mc_convergence", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unique_average_by_s(np, s_param, values):
    s = np.asarray(s_param, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    order = np.argsort(s)
    s_sorted = s[order]
    v_sorted = values[order]
    uniq, inv = np.unique(np.round(s_sorted, 14), return_inverse=True)
    if uniq.size == s_sorted.size:
        return s_sorted, v_sorted
    accum = np.zeros(uniq.size, dtype=float)
    counts = np.zeros(uniq.size, dtype=float)
    np.add.at(accum, inv, v_sorted)
    np.add.at(counts, inv, 1.0)
    return uniq, accum / np.maximum(counts, 1.0)


def _parametric_l2_error(np, conv, s_param, slip_phys, interior: float):
    s, slip = _unique_average_by_s(np, s_param, slip_phys)
    s_aug = np.concatenate(([-1.0], s, [1.0]))
    slip_aug = np.concatenate(([0.0], slip, [0.0]))
    order = np.argsort(s_aug)
    s_aug = s_aug[order]
    slip_aug = slip_aug[order]
    s_aug, slip_aug = _unique_average_by_s(np, s_aug, slip_aug)

    a, b = -float(interior), float(interior)
    mask = (s_aug > a) & (s_aug < b)
    s_eval = np.concatenate(([a], s_aug[mask], [b]))
    slip_eval = np.interp(s_eval, s_aug, slip_aug)
    anal_eval = conv.slip_max_anal * np.sqrt(np.clip(1.0 - s_eval ** 2, 0.0, None))
    err = np.abs(slip_eval) - np.abs(anal_eval)
    err_l2 = float(np.sqrt(np.trapezoid(err * err, s_eval)))
    ref_l2 = float(np.sqrt(np.trapezoid(anal_eval * anal_eval, s_eval)))
    rel_l2 = err_l2 / max(ref_l2, 1.0e-30)
    return err_l2, ref_l2, rel_l2, s_eval, slip_eval, anal_eval


def _fit_rate(np, rows: list[dict], error_key: str) -> tuple[float, float]:
    if len(rows) < 2:
        return math.nan, math.nan
    h = np.log([float(row["h_eff"]) for row in rows])
    err = np.log([float(row[error_key]) for row in rows])
    slope, intercept = np.polyfit(h, err, 1)
    pred = slope * h + intercept
    ss_res = float(np.sum((err - pred) ** 2))
    ss_tot = float(np.sum((err - np.mean(err)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    return float(slope), float(r2)


def _endpoint_rate(rows: list[dict], error_key: str) -> float:
    if len(rows) < 2:
        return math.nan
    first = rows[0]
    last = rows[-1]
    return float(
        math.log(float(first[error_key]) / float(last[error_key]))
        / math.log(float(first["h_eff"]) / float(last["h_eff"]))
    )


def _configure_case(conv, cfg: CaseConfig, audit_dir: Path | None) -> None:
    conv.ELEMENT_TYPE = cfg.element_type
    conv.MESH_SOURCE = cfg.mesh_source
    conv.USE_SBM = bool(cfg.use_sbm)
    conv.CONFORMING_CRACK_MESH = not bool(cfg.use_sbm)
    conv.INCLUDE_SBM = False
    conv.INCLUDE_TAYLOR = True
    conv.INCLUDE_TAYLOR_TEST = False
    conv.INCLUDE_HESSIAN = False
    conv.TAYLOR_METHOD = "nodal"
    conv.LUMPED_COUPLING = "consistent"
    conv.AUDIT_DIR = audit_dir


def _write_slip_profile(np, conv, case_dir: Path, case_name: str, result, interior: float) -> None:
    _, _, _, s_eval, slip_eval, anal_eval = _parametric_l2_error(
        np, conv, result.s_param, result.slip_phys, interior
    )
    path = case_dir / f"slip_profile_N{result.n_elem_target}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "n_elem_target",
                "s_param",
                "slip_interp_m",
                "abs_slip_interp_m",
                "analytical_slip_m",
                "error_m",
            ],
        )
        writer.writeheader()
        for s, slip, anal in zip(s_eval, slip_eval, anal_eval):
            writer.writerow(
                {
                    "case": case_name,
                    "n_elem_target": result.n_elem_target,
                    "s_param": f"{float(s):.16e}",
                    "slip_interp_m": f"{float(slip):.16e}",
                    "abs_slip_interp_m": f"{abs(float(slip)):.16e}",
                    "analytical_slip_m": f"{float(anal):.16e}",
                    "error_m": f"{abs(float(slip)) - float(anal):.16e}",
                }
            )


def _plot_slip_case(np, plt, conv, rows: list[dict], results: list, case_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    s_dense = np.linspace(-1.0, 1.0, 501)
    anal = conv.slip_max_anal * np.sqrt(np.clip(1.0 - s_dense ** 2, 0.0, None))
    ax.plot(s_dense, anal, "k--", lw=1.7, label="analytical")
    colors = plt.cm.viridis(np.linspace(0.06, 0.94, len(results)))
    for color, result in zip(colors, results):
        ax.plot(
            result.s_param_with_tips,
            np.abs(result.slip_with_tips),
            marker="o",
            ms=3.0,
            lw=1.0,
            color=color,
            label=f"N={result.n_elem_target}, n_c={result.n_c}",
        )
    ax.set_xlabel(r"true-crack parametric coordinate $s=\xi/c$")
    ax.set_ylabel(r"total slip magnitude $|[\![u_t]\!]|$ (m)")
    ax.set_title(title)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(case_dir / "slip_vs_analytical.png", dpi=180)
    plt.close(fig)


def _plot_convergence_case(np, plt, rows: list[dict], case_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    h = np.asarray([float(row["h_eff"]) for row in rows])
    err = np.asarray([float(row["err_l2_parametric"]) for row in rows])
    ax.loglog(h, err, "o-", ms=6, lw=1.4, label="parametric L2")
    if len(rows) >= 2:
        p, r2 = _fit_rate(np, rows, "err_l2_parametric")
        h_fit = np.asarray([float(np.min(h)), float(np.max(h))])
        intercept = np.polyfit(np.log(h), np.log(err), 1)[1]
        ax.loglog(h_fit, np.exp(intercept) * h_fit ** p, "--", lw=1.1, label=f"LS p={p:.2f}, R2={r2:.2f}")
    ax.invert_xaxis()
    ax.set_xlabel(r"$h_{\rm eff}$")
    ax.set_ylabel(r"$\left(\int |e(s)|^2\,ds\right)^{1/2}$ (m)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(case_dir / "convergence_l2_parametric.png", dpi=180)
    plt.close(fig)


def _plot_all_convergence(np, plt, all_rows: list[dict], rates: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    cases = []
    for row in all_rows:
        if row["case"] not in cases:
            cases.append(row["case"])
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(cases), 1)))
    rate_by_case = {row["case"]: row for row in rates}
    for color, case in zip(colors, cases):
        rows = [row for row in all_rows if row["case"] == case]
        rows = sorted(rows, key=lambda row: int(row["n_elem_target"]))
        label = case
        if case in rate_by_case and math.isfinite(float(rate_by_case[case]["ls_rate_parametric"])):
            label += f" p={float(rate_by_case[case]['ls_rate_parametric']):.2f}"
        ax.loglog(
            [float(row["h_eff"]) for row in rows],
            [float(row["err_l2_parametric"]) for row in rows],
            "o-",
            lw=1.4,
            ms=5.5,
            color=color,
            label=label,
        )
    ax.invert_xaxis()
    ax.set_xlabel(r"$h_{\rm eff}$")
    ax.set_ylabel(r"$\left(\int |e(s)|^2\,ds\right)^{1/2}$ (m)")
    ax.set_title("Mohr-Coulomb slip convergence on true-crack parametric coordinate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_rates_parametric.png", dpi=180)
    plt.close(fig)


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cleanup_petsc() -> None:
    gc.collect()
    try:
        from petsc4py import PETSc

        PETSc.garbage_cleanup()
    except Exception:
        pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=_split_csv_ints,
        default=_split_csv_ints("20,30,40,50,60,70,80"),
        help="Comma-separated N_ELEM targets. Cases and levels run sequentially.",
    )
    parser.add_argument(
        "--cases",
        type=_split_csv_strings,
        default=list(CASE_CONFIGS),
        help=f"Comma-separated cases. Available: {','.join(CASE_CONFIGS)}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/_audit_mohr_coulomb_convergence_test"),
        help="Output directory for CSVs, plots, and per-case audit sidecars.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Thread count exported via OMP/BLAS env vars before importing PETSc/Numpy.",
    )
    parser.add_argument(
        "--interior",
        type=float,
        default=0.95,
        help="Interior window |s| < value used for L2 errors.",
    )
    parser.add_argument(
        "--no-audits",
        action="store_true",
        help="Disable per-level contact audit sidecars from the convergence module.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort the whole sweep if any level fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sequential plan and exit without importing solver modules.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    unknown = [case for case in args.cases if case not in CASE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")

    rank = _mpi_rank_from_env()
    if rank != 0:
        print(
            f"[rank {rank}] exiting: this sequential runner uses PETSc COMM_SELF; "
            "run it with one process and --cores for threaded BLAS/MUMPS.",
            flush=True,
        )
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    _configure_thread_env(args.cores)

    plan = {
        "levels": args.levels,
        "cases": args.cases,
        "cores": int(args.cores),
        "interior": float(args.interior),
        "sequential": True,
        "petsc_note": "Solver path uses PETSc/MUMPS with PETSc COMM_SELF; --cores sets thread env vars.",
    }
    (args.out / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    repo_root = Path(__file__).resolve().parents[1]
    conv = _load_convergence_module(repo_root)

    all_rows: list[dict] = []
    failure_rows: list[dict] = []
    case_results: dict[str, list] = {}

    for case_name in args.cases:
        cfg = CASE_CONFIGS[case_name]
        case_dir = args.out / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = None if args.no_audits else case_dir / "audits"
        if audit_dir is not None:
            audit_dir.mkdir(parents=True, exist_ok=True)
        _configure_case(conv, cfg, audit_dir)
        results = []
        case_rows = []

        print(f"\n=== case: {case_name} ({cfg.label}) ===", flush=True)
        for level_index, n_elem in enumerate(args.levels):
            try:
                result = conv.run_level(int(n_elem))
                results.append(result)

                err_l2, ref_l2, rel_l2, _, _, _ = _parametric_l2_error(
                    np, conv, result.s_param, result.slip_phys, args.interior
                )
                prev = case_rows[-1] if case_rows else None
                adjacent_rate = ""
                if prev is not None and result.h_eff < float(prev["h_eff"]):
                    adjacent_rate = (
                        math.log(float(prev["err_l2_parametric"]) / err_l2)
                        / math.log(float(prev["h_eff"]) / result.h_eff)
                    )
                row = {
                    "case": case_name,
                    "case_label": cfg.label,
                    "level_index": level_index,
                    "n_elem_target": result.n_elem_target,
                    "n_elem_actual": result.n_elem_actual,
                    "n_contact_nodes": result.n_c,
                    "h_eff": f"{result.h_eff:.16e}",
                    "err_l2_parametric": f"{err_l2:.16e}",
                    "ref_l2_parametric": f"{ref_l2:.16e}",
                    "rel_l2_parametric": f"{rel_l2:.16e}",
                    "adjacent_rate_parametric": "" if adjacent_rate == "" else f"{adjacent_rate:.16e}",
                    "err_l2_physical_arclength": f"{result.err_l2:.16e}",
                    "ref_l2_physical_arclength": f"{result.ref_l2:.16e}",
                    "rel_l2_physical_arclength": f"{result.rel_l2:.16e}",
                    "gap_shift_rel": f"{result.gap_shift_rel:.16e}",
                    "soc_res_inf": f"{result.soc_res_inf:.16e}",
                    "max_friction_power_pos": f"{result.max_friction_power_pos:.16e}",
                    "min_gap": f"{result.min_gap:.16e}",
                    "final_cone_margin_abs": f"{result.final_cone_margin_abs:.16e}",
                    "contact_ok": int(result.contact_ok),
                    "shifted_true_crack_hatted": int(cfg.use_sbm),
                    "rotated_to_normal_tangential": 1,
                    "element_type": cfg.element_type,
                    "mesh_source": cfg.mesh_source,
                    "t_build_s": f"{result.t_build:.6e}",
                    "t_solve_s": f"{result.t_solve:.6e}",
                    "t_total_s": f"{result.t_total:.6e}",
                    "n_acc": result.n_acc,
                    "n_rej": result.n_rej,
                }
                case_rows.append(row)
                all_rows.append(row)
                _write_slip_profile(np, conv, case_dir, case_name, result, args.interior)

                _write_rows(case_dir / "l2_errors.csv", case_rows, L2_FIELDNAMES)
                _write_rows(args.out / "l2_errors.csv", all_rows, L2_FIELDNAMES)
                _plot_slip_case(np, plt, conv, case_rows, results, case_dir, cfg.label)
                _plot_convergence_case(np, plt, case_rows, case_dir, cfg.label)
            except Exception as exc:
                failure = {
                    "case": case_name,
                    "n_elem_target": int(n_elem),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failure_rows.append(failure)
                _write_rows(args.out / "failures.csv", failure_rows, FAILURE_FIELDNAMES)
                print(f"[failed] {failure}", flush=True)
                if args.stop_on_failure:
                    raise
            finally:
                _cleanup_petsc()

        case_results[case_name] = results

    rate_rows = []
    for case_name in args.cases:
        rows = [row for row in all_rows if row["case"] == case_name]
        rows = sorted(rows, key=lambda row: int(row["n_elem_target"]))
        p, r2 = _fit_rate(np, rows, "err_l2_parametric")
        endpoint = _endpoint_rate(rows, "err_l2_parametric")
        rate_rows.append(
            {
                "case": case_name,
                "n_levels": len(rows),
                "ls_rate_parametric": f"{p:.16e}",
                "ls_r2_parametric": f"{r2:.16e}",
                "endpoint_rate_parametric": f"{endpoint:.16e}",
                "first_N": rows[0]["n_elem_target"] if rows else "",
                "last_N": rows[-1]["n_elem_target"] if rows else "",
            }
        )

    _write_rows(args.out / "convergence_rates.csv", rate_rows, RATE_FIELDNAMES)
    if all_rows:
        _plot_all_convergence(np, plt, all_rows, rate_rows, args.out)

    print(f"\noutputs written to: {args.out.resolve()}", flush=True)
    return 0


L2_FIELDNAMES = [
    "case",
    "case_label",
    "level_index",
    "n_elem_target",
    "n_elem_actual",
    "n_contact_nodes",
    "h_eff",
    "err_l2_parametric",
    "ref_l2_parametric",
    "rel_l2_parametric",
    "adjacent_rate_parametric",
    "err_l2_physical_arclength",
    "ref_l2_physical_arclength",
    "rel_l2_physical_arclength",
    "gap_shift_rel",
    "soc_res_inf",
    "max_friction_power_pos",
    "min_gap",
    "final_cone_margin_abs",
    "contact_ok",
    "shifted_true_crack_hatted",
    "rotated_to_normal_tangential",
    "element_type",
    "mesh_source",
    "t_build_s",
    "t_solve_s",
    "t_total_s",
    "n_acc",
    "n_rej",
]


RATE_FIELDNAMES = [
    "case",
    "n_levels",
    "ls_rate_parametric",
    "ls_r2_parametric",
    "endpoint_rate_parametric",
    "first_N",
    "last_N",
]


FAILURE_FIELDNAMES = ["case", "n_elem_target", "error_type", "error"]


if __name__ == "__main__":
    raise SystemExit(main())
