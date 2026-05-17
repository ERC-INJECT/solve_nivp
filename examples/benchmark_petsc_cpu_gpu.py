"""Benchmark PETSc CPU vs GPU backends on the same sparse linear solve.

This script exercises the solver's PETSc path directly with a fixed sparse
matrix sequence so that we can compare:

* CPU PETSc: ``gmres + jacobi``
* GPU PETSc: ``gmres + jacobi + aijcusparse/cuda``

It uses a 2-D five-point Laplacian shifted by the identity. The matrix stays
constant across solves; only the right-hand side changes. That keeps the
benchmark simple while still measuring host-side CSR handling and PETSc object
reuse in the current implementation.

Example
-------
    /home/david/enter/envs/fem-env/bin/python examples/benchmark_petsc_cpu_gpu.py \
        --n-side 64 --repeats 8 --warmup 2
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import scipy.sparse as sp

from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection


CPU_PETSC_OPTS = {
    "ksp_type": "gmres",
    "pc_type": "jacobi",
}

GPU_PETSC_OPTS = {
    "ksp_type": "gmres",
    "pc_type": "jacobi",
    "mat_type": "aijcusparse",
    "vec_type": "cuda",
}


def build_shifted_laplacian(n_side: int, shift: float) -> sp.csr_matrix:
    """Return ``I + shift * L`` for the 2-D five-point Laplacian."""
    e = np.ones(n_side, dtype=float)
    t = sp.diags([-e, 2.0 * e, -e], [-1, 0, 1], shape=(n_side, n_side), format="csr")
    lap = sp.kronsum(t, t, format="csr")
    n = n_side * n_side
    return (sp.eye(n, format="csr") + float(shift) * lap).tocsr()


def make_rhs_sequence(n: int, repeats: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n) for _ in range(repeats)]


def time_petsc_backend(
    J: sp.csr_matrix,
    rhs_list: list[np.ndarray],
    petsc_options: dict,
    *,
    petsc_comm: str = "world",
) -> dict:
    """Time repeated PETSc solves on one backend."""
    solver = ImplicitEquationSolver(
        method="semismooth_newton",
        proj=IdentityProjection(),
        linear_solver="petsc",
        petsc_comm=petsc_comm,
        petsc_options=dict(petsc_options),
    )

    solve_times = []
    solutions = []
    warning_messages = []

    for rhs in rhs_list:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            x, ok = solver._solve_with_petsc(J, rhs)
            elapsed = time.perf_counter() - t0
        if not ok:
            raise RuntimeError(f"PETSc solve failed for options {petsc_options}")

        solve_times.append(elapsed)
        solutions.append(x)
        warning_messages.extend(str(w.message) for w in caught)

    unique_warnings = list(dict.fromkeys(warning_messages))
    return {
        "requested_options": dict(petsc_options),
        "use_gpu": bool(getattr(solver, "_petsc_use_gpu", False)),
        "effective_mat_type": getattr(solver, "_petsc_effective_mat_type", None),
        "effective_vec_type": getattr(solver, "_petsc_effective_vec_type", None),
        "solve_times_s": solve_times,
        "solutions": solutions,
        "warnings": unique_warnings,
    }


def summarise_timings(name: str, result: dict, warmup: int) -> dict:
    times = np.asarray(result["solve_times_s"], dtype=float)
    steady = times[min(warmup, len(times)):] if len(times) > 0 else times
    if steady.size == 0:
        steady = times
    return {
        "backend": name,
        "use_gpu": result["use_gpu"],
        "effective_mat_type": result["effective_mat_type"],
        "effective_vec_type": result["effective_vec_type"],
        "first_solve_s": float(times[0]),
        "mean_steady_s": float(np.mean(steady)),
        "median_steady_s": float(np.median(steady)),
        "min_steady_s": float(np.min(steady)),
        "max_steady_s": float(np.max(steady)),
        "warnings": result["warnings"],
    }


def print_summary(problem: dict, cpu_summary: dict, gpu_summary: dict, max_abs_diff: float) -> None:
    print("Problem")
    print(json.dumps(problem, indent=2))
    print()
    print("CPU PETSc")
    print(json.dumps(cpu_summary, indent=2))
    print()
    print("GPU PETSc")
    print(json.dumps(gpu_summary, indent=2))
    print()
    print("Validation")
    print(json.dumps({"max_abs_solution_difference": max_abs_diff}, indent=2))
    print()
    if cpu_summary["mean_steady_s"] > 0.0 and gpu_summary["mean_steady_s"] > 0.0:
        speedup = cpu_summary["mean_steady_s"] / gpu_summary["mean_steady_s"]
        print("Derived")
        print(json.dumps({"gpu_speedup_vs_cpu_steady": speedup}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-side", type=int, default=64, help="Grid side length for the 2-D Laplacian.")
    parser.add_argument("--shift", type=float, default=0.1, help="Shift in I + shift * L.")
    parser.add_argument("--repeats", type=int, default=8, help="Number of repeated solves per backend.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of initial solves to exclude from steady-state statistics.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for right-hand sides.")
    args = parser.parse_args()

    J = build_shifted_laplacian(args.n_side, args.shift)
    rhs_list = make_rhs_sequence(J.shape[0], args.repeats, args.seed)

    cpu_result = time_petsc_backend(J, rhs_list, CPU_PETSC_OPTS)
    gpu_result = time_petsc_backend(J, rhs_list, GPU_PETSC_OPTS)

    cpu_summary = summarise_timings("cpu", cpu_result, args.warmup)
    gpu_summary = summarise_timings("gpu", gpu_result, args.warmup)

    max_abs_diff = max(
        float(np.max(np.abs(x_cpu - x_gpu)))
        for x_cpu, x_gpu in zip(cpu_result["solutions"], gpu_result["solutions"])
    )

    problem = {
        "matrix": "I + shift * 2D five-point Laplacian",
        "n_side": args.n_side,
        "dofs": int(J.shape[0]),
        "nnz": int(J.nnz),
        "shift": float(args.shift),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
    }
    print_summary(problem, cpu_summary, gpu_summary, max_abs_diff)


if __name__ == "__main__":
    main()
