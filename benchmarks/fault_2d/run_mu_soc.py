"""2D slip-weakening fault benchmark using MuScaledSOCProjection.

Same physics as baseline_coulomb.py but reformulates the 1D tangential
friction constraint as a degenerate 2D second-order cone per node by
augmenting the state with a held-constant normal component s_i = sigma_n_i.

State layout: y = [v (N), u (N), S (N), sN (N)]
  - ds_i/dt = 0, s_i(0) = sigma_n_i
  - SOC block k = (idx_sN = 3N+k, [idx_v = k])
  - get_mu(y) returns mu_res*(1 - (dmu/mu_res)*exp(-S/dc))  per node
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
from solve_nivp.projections import MuScaledSOCProjection
from plants.faults import strikeslip

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_problem(Nx: int, Nz: int):
    os.chdir(_RL_DIR)
    fault = strikeslip.qs_strikeslip_fault(
        zdepth=3, xlength=3, Nz=Nz, Nx=Nx,
        G=30000., rho=2.5e-3, zeta=0.8 / 3,
        Ks_path="./Data/", gamma_s=25., gamma_w=10.,
        sigma_ref=100., depth_ini=0., vinf=3.171e-10,
        Dmu_estimate=.5,
    )
    MA, KS, ES, SIGMA_N, VINF_raw = fault.get_plant()

    N = fault.N
    VINF = VINF_raw * np.ones(N) * 0

    # Augmented descriptor: block_diag(MA, I, I, I)
    I_N = sp.eye(N, format="csr")
    A = sp.block_diag([sp.csr_matrix(MA), I_N, I_N, I_N], format="csr")

    component_slices = [
        slice(0, N),
        slice(N, 2 * N),
        slice(2 * N, 3 * N),
        slice(3 * N, 4 * N),
    ]

    KS_dense = np.ascontiguousarray(KS, dtype=np.float64)
    ES_dense = np.ascontiguousarray(ES, dtype=np.float64)
    VINF_vec = np.ascontiguousarray(VINF, dtype=np.float64)
    SIGMA_N_vec = np.ascontiguousarray(SIGMA_N, dtype=np.float64)

    rhs_buffer = np.zeros(4 * N, dtype=np.float64)
    rhs_jac_buffer = np.zeros((4 * N, 4 * N), dtype=np.float64)

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
            out[3 * n + i] = 0.0

    def rhs(t, y):
        _rhs_kernel(y, KS_dense, ES_dense, VINF_vec, rhs_buffer)
        return rhs_buffer.copy()

    def rhs_jac(t, y, Fk_val=None):
        jac = rhs_jac_buffer
        jac.fill(0.0)
        jac[:N, :N] = -ES_dense
        jac[:N, N:2 * N] = -KS_dense
        jac[N + np.arange(N), np.arange(N)] = 1.0
        slip_signs = np.sign(y[:N]).astype(np.float64)
        jac[2 * N + np.arange(N), np.arange(N)] = slip_signs
        return jac.copy()

    DMU = -0.1
    DC = 100.0 / fault.Dscale
    MU_RES = 0.5

    @njit(cache=True)
    def _mu_of_slip(slip, mu_res, dmu, dc, out):
        n = slip.shape[0]
        for i in range(n):
            out[i] = mu_res * (1.0 - (dmu / mu_res) * math.exp(-slip[i] / dc))

    mu_buffer = np.zeros(N, dtype=np.float64)

    def get_mu(y):
        slip = y[2 * N:3 * N]
        _mu_of_slip(slip, MU_RES, DMU, DC, mu_buffer)
        return mu_buffer.copy()

    # Initial condition: reuse Coulomb static-friction seed, then pad with sigma_n
    y0 = np.zeros(4 * N)
    friction_force = MU_RES * SIGMA_N_vec  # mu(S=0)*sigma_n
    uc = -np.linalg.solve(KS, friction_force)
    u0 = uc * (1 + 1e-5)
    y0[N:2 * N] = u0
    y0[3 * N:4 * N] = SIGMA_N_vec

    # SOC blocks: per node, normal idx in [3N, 4N), tangential idx in [0, N)
    blocks = [(3 * N + i, [i]) for i in range(N)]

    projection = MuScaledSOCProjection(
        blocks=blocks,
        get_mu=get_mu,
        component_slices=component_slices,
    )

    return {
        "fault": fault,
        "A": A,
        "rhs": rhs,
        "rhs_jac": rhs_jac,
        "y0": y0,
        "N": N,
        "component_slices": component_slices,
        "projection": projection,
    }


def run(problem):
    fault = problem["fault"]
    N = problem["N"]

    solver_opts = dict(
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
    t, y, h, fk, info = solve_nivp.solve_nivp(
        fun=problem["rhs"],
        t_span=t_span,
        y0=problem["y0"],
        method="composite",
        projection=problem["projection"],
        solver="semismooth_newton",
        projection_opts={},
        solver_opts=solver_opts,
        adaptive=True,
        adaptive_opts=adaptive_opts,
        rtol=1e-1,
        h0=30 / 4 * fault.second / fault.Tscale,
        component_slices=problem["component_slices"],
        verbose=True,
        A=problem["A"],
    )
    wall = time.time() - start
    print(f"mu_soc solve complete in {wall:.3f} s")
    return {"t": t, "y": y, "h": h, "wall_time": wall}


def save_and_plot(result, problem, label: str):
    fault = problem["fault"]
    N = problem["N"]
    t_s = result["t"] * fault.Tscale / fault.second
    mean_v = np.mean(result["y"][:, :N], axis=1) * fault.Vscale

    np.savez(
        RESULTS_DIR / f"{label}.npz",
        t_seconds=t_s,
        mean_velocity=mean_v,
        wall_time=result["wall_time"],
        N=N,
    )

    # Overlay against Coulomb baseline if available
    base_path = RESULTS_DIR / f"coulomb_{Nx}x{Nz}.npz"
    fig, ax = plt.subplots(figsize=(8, 5))
    if base_path.exists():
        base = np.load(base_path)
        ax.plot(base["t_seconds"], base["mean_velocity"], "-o", color="C0",
                markersize=4, label=f"coulomb (wall={float(base['wall_time']):.1f}s)")
    ax.plot(t_s, mean_v, "--s", color="C1", markersize=4,
            label=f"mu_soc (wall={result['wall_time']:.1f}s)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean slip rate [m/s]")
    ax.set_title(f"2D fault — {label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{label}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / (label + '.npz')} and .png")


if __name__ == "__main__":
    Nx = Nz = 10
    problem = build_problem(Nx=Nx, Nz=Nz)
    result = run(problem)
    save_and_plot(result, problem, label=f"mu_soc_{Nx}x{Nz}")
