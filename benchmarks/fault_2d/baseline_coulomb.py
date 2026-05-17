"""Baseline 2D slip-weakening fault benchmark with Coulomb projection.

Replicates the notebook setup from RL_Adaption/2D_FAULT/2d_Fault.ipynb verbatim
and saves the mean slip-rate time series for later cross-backend validation.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numba import njit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
_RL_DIR = _REPO_ROOT / "RL_Adaption" / "2D_FAULT"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))

import solve_nivp
from plants.faults import strikeslip

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_problem(Nx: int = 50, Nz: int = 50):
    os.chdir(_RL_DIR)
    fault = strikeslip.qs_strikeslip_fault(
        zdepth=3, xlength=3, Nz=Nz, Nx=Nx,
        G=30000., rho=2.5e-3, zeta=0.8 / 3,
        Ks_path="./Data/", gamma_s=25., gamma_w=10.,
        sigma_ref=100., depth_ini=0., vinf=3.171e-10,
        Dmu_estimate=.5,
    )
    MA, KS, ES, SIGMA_N, VINF_raw = fault.get_plant()

    N_DOFS = fault.N
    VINF = VINF_raw * np.ones(N_DOFS) * 0

    I_N = sp.eye(N_DOFS, format="csr")
    A = sp.block_diag([sp.csr_matrix(MA), I_N, I_N], format="csr")

    component_slices = [
        slice(0, N_DOFS),
        slice(N_DOFS, 2 * N_DOFS),
        slice(2 * N_DOFS, 3 * N_DOFS),
    ]

    KS_dense = np.ascontiguousarray(KS, dtype=np.float64)
    ES_dense = np.ascontiguousarray(ES, dtype=np.float64)
    VINF_vec = np.ascontiguousarray(VINF, dtype=np.float64)
    SIGMA_N_vec = np.ascontiguousarray(SIGMA_N, dtype=np.float64)

    rhs_buffer = np.empty(3 * N_DOFS, dtype=np.float64)
    rhs_jac_buffer = np.zeros((3 * N_DOFS, 3 * N_DOFS), dtype=np.float64)
    con_force_buffer = np.zeros(3 * N_DOFS, dtype=np.float64)
    con_force_jac_buffer = np.zeros((3 * N_DOFS, 3 * N_DOFS), dtype=np.float64)

    @njit(cache=True)
    def _rhs_kernel(y, KS, ES, VINF, out):
        n = VINF.shape[0]
        for i in range(n):
            sum_ku = 0.0
            sum_ev = 0.0
            yi = y[i]
            for j in range(n):
                sum_ku += KS[i, j] * y[n + j]
                sum_ev += ES[i, j] * (y[j] - VINF[j])
            out[i] = -(sum_ku + sum_ev)
            out[n + i] = yi - VINF[i]
            out[2 * n + i] = yi if yi >= 0.0 else -yi

    def rhs(t, y):
        _rhs_kernel(y, KS_dense, ES_dense, VINF_vec, rhs_buffer)
        return rhs_buffer.copy()

    def rhs_jac(t, y, Fk_val=None):
        n = N_DOFS
        jac = rhs_jac_buffer
        jac.fill(0.0)
        jac[:n, :n] = -ES_dense
        jac[:n, n:2 * n] = -KS_dense
        jac[n + np.arange(n), np.arange(n)] = 1.0
        slip_signs = np.sign(y[:n]).astype(np.float64)
        jac[2 * n + np.arange(n), np.arange(n)] = slip_signs
        return jac.copy()

    DMU = -0.1
    DC = 100.0 / fault.Dscale
    MU_RES = 0.5

    @njit(cache=True)
    def _con_force_kernel(state, sigma_n, mu_res, dmu, dc, out):
        n = sigma_n.shape[0]
        for i in range(n):
            slip_i = state[2 * n + i]
            mu_val = mu_res * (1.0 - (dmu / mu_res) * math.exp(-slip_i / dc))
            out[i] = mu_val * sigma_n[i]

    @njit(cache=True)
    def _con_force_jac_kernel(state, sigma_n, dmu, dc, out):
        n = sigma_n.shape[0]
        for i in range(n):
            slip_i = state[2 * n + i]
            dmu_dslip = (dmu / dc) * math.exp(-slip_i / dc)
            out[i, 2 * n + i] = sigma_n[i] * dmu_dslip

    def con_force(state, fk=None):
        con_force_buffer.fill(0.0)
        _con_force_kernel(state, SIGMA_N_vec, MU_RES, DMU, DC, con_force_buffer)
        return con_force_buffer.copy()

    def con_force_jacobian(state, t=None, Fk_val=None):
        con_force_jac_buffer.fill(0.0)
        _con_force_jac_kernel(state, SIGMA_N_vec, DMU, DC, con_force_jac_buffer)
        return con_force_jac_buffer.copy()

    y0 = np.zeros(3 * N_DOFS)
    friction_force = con_force(y0)
    uc = -np.linalg.solve(KS, friction_force[:N_DOFS])
    u0 = uc * (1 + 1e-5)
    y0[N_DOFS:2 * N_DOFS] = u0

    return {
        "fault": fault,
        "A": A,
        "rhs": rhs,
        "rhs_jac": rhs_jac,
        "con_force": con_force,
        "con_force_jacobian": con_force_jacobian,
        "y0": y0,
        "N_DOFS": N_DOFS,
        "component_slices": component_slices,
    }


def run_coulomb(problem):
    fault = problem["fault"]
    N_DOFS = problem["N_DOFS"]

    projection_opts = {
        "con_force_func": problem["con_force"],
        "rhok": np.ones(N_DOFS, dtype=float),
        "component_slices": problem["component_slices"],
        "constraint_indices": np.arange(N_DOFS, dtype=np.int32),
        "use_numba": True,
    }

    solver_opts_ssn = dict(
        tol=1e-8,
        max_iter=200,
        rhs_jac=problem["rhs_jac"],
        vi_strict_block_lipschitz=False,
        vi_max_block_adjust_iters=5,
        globalization="line_search",
    )

    adaptive_opts = dict(
        h0=5e-2,
        h_min=1e-7,
        h_down=0.6,
        h_up=1.8,
        method_order=1,
        skip_error_indices=[],
        controller="h211b",
        b_param=4.0,
        mode="ratio",
    )

    tmax = 30 * fault.second / fault.Tscale
    t_span = (0.0, tmax)

    start = time.time()
    t_vals, y_vals, h_vals, fk_vals, info = solve_nivp.solve_nivp(
        fun=problem["rhs"],
        t_span=t_span,
        y0=problem["y0"],
        method="composite",
        projection="coulomb",
        solver="semismooth_newton",
        projection_opts=projection_opts,
        solver_opts=solver_opts_ssn,
        adaptive=True,
        adaptive_opts=adaptive_opts,
        rtol=1e-1,
        h0=30 / 4 * fault.second / fault.Tscale,
        component_slices=problem["component_slices"],
        verbose=True,
        A=problem["A"],
    )
    wall = time.time() - start
    print(f"Coulomb solve complete in {wall:.3f} s")

    return {
        "t": t_vals,
        "y": y_vals,
        "h": h_vals,
        "wall_time": wall,
    }


def save_reference(result, problem, label: str):
    fault = problem["fault"]
    N_DOFS = problem["N_DOFS"]
    t_s = result["t"] * fault.Tscale / fault.second
    mean_v = np.mean(result["y"][:, :N_DOFS], axis=1) * fault.Vscale
    out_path = RESULTS_DIR / f"{label}.npz"
    np.savez(
        out_path,
        t_seconds=t_s,
        mean_velocity=mean_v,
        t_nd=result["t"],
        h=result["h"],
        wall_time=result["wall_time"],
        N_DOFS=N_DOFS,
    )
    print(f"Saved {out_path}  (steps={len(t_s)}, wall={result['wall_time']:.3f} s)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_s, mean_v, "-o", color="C0", markersize=4, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean slip rate [m/s]")
    ax.set_title(f"2D fault — {label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"{label}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")
    return out_path


if __name__ == "__main__":
    problem = build_problem(Nx=10, Nz=10)
    result = run_coulomb(problem)
    save_reference(result, problem, label="coulomb_10x10")
