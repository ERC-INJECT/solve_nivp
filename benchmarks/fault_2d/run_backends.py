"""Build one FrictionOnlyProblem and overlay mean-velocity traces across backends."""

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
_RL_DIR = _REPO_ROOT / "RL_Adaption" / "2D_FAULT"
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))

import solve_nivp
from plants.faults import strikeslip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import FrictionOnlyProblem

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_friction_problem(Nx: int, Nz: int) -> FrictionOnlyProblem:
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

    I_N = sp.eye(N, format="csr")
    A = sp.block_diag([sp.csr_matrix(MA), I_N, I_N], format="csr")

    component_slices = [
        slice(0, N),
        slice(N, 2 * N),
        slice(2 * N, 3 * N),
    ]

    KS_dense = np.ascontiguousarray(KS, dtype=np.float64)
    ES_dense = np.ascontiguousarray(ES, dtype=np.float64)
    VINF_vec = np.ascontiguousarray(VINF, dtype=np.float64)
    SIGMA_N_vec = np.ascontiguousarray(SIGMA_N, dtype=np.float64)

    rhs_buffer = np.empty(3 * N, dtype=np.float64)
    rhs_jac_buffer = np.zeros((3 * N, 3 * N), dtype=np.float64)

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

    def rhs_smooth(t, y):
        _rhs_kernel(y, KS_dense, ES_dense, VINF_vec, rhs_buffer)
        return rhs_buffer.copy()

    def rhs_jac(t, y, Fk_val=None):
        jac = rhs_jac_buffer
        jac.fill(0.0)
        jac[:N, :N] = -ES_dense
        jac[:N, N:2 * N] = -KS_dense
        jac[N + np.arange(N), np.arange(N)] = 1.0
        jac[2 * N + np.arange(N), np.arange(N)] = np.sign(y[:N]).astype(np.float64)
        return jac.copy()

    DMU = -0.1
    DC = 100.0 / fault.Dscale
    MU_RES = 0.5

    @njit(cache=True)
    def _mu_kernel(slip, mu_res, dmu, dc, out):
        n = slip.shape[0]
        for i in range(n):
            out[i] = mu_res * (1.0 - (dmu / mu_res) * math.exp(-slip[i] / dc))

    mu_buf = np.zeros(N, dtype=np.float64)

    def mu_of_slip(slip):
        _mu_kernel(np.ascontiguousarray(slip, dtype=np.float64),
                   MU_RES, DMU, DC, mu_buf)
        return mu_buf.copy()

    y0 = np.zeros(3 * N)
    mu_at_zero = mu_of_slip(np.zeros(N))
    friction_force = mu_at_zero * SIGMA_N_vec
    uc = -np.linalg.solve(KS, friction_force)
    u0 = uc * (1 + 1e-5)
    y0[N:2 * N] = u0

    problem = FrictionOnlyProblem(
        rhs_smooth_base=rhs_smooth,
        rhs_jac_base=rhs_jac,
        A_base=A,
        y0_base=y0,
        N=N,
        mu_of_slip=mu_of_slip,
        sigma_n=SIGMA_N_vec,
        component_slices_base=component_slices,
    )
    return problem, fault


def run_bundle(bundle, fault, label):
    tmax = 30 * fault.second / fault.Tscale
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
    start = time.time()
    t, y, h, fk, info = solve_nivp.solve_nivp(
        fun=bundle.fun,
        t_span=(0.0, tmax),
        y0=bundle.y0,
        method="composite",
        projection=bundle.projection,
        solver="semismooth_newton",
        projection_opts=bundle.projection_opts,
        solver_opts=bundle.solver_opts,
        adaptive=True,
        adaptive_opts=adaptive_opts,
        rtol=1e-1,
        h0=30 / 4 * fault.second / fault.Tscale,
        component_slices=bundle.component_slices,
        verbose=False,
        A=bundle.A,
    )
    wall = time.time() - start
    t_s = t * fault.Tscale / fault.second
    mean_v = np.mean(y[:, :bundle.extract_velocity(y[0]).size], axis=1) * fault.Vscale
    print(f"[{label}] wall={wall:.2f}s  steps={len(t)}")
    return dict(t=t_s, mean_v=mean_v, wall=wall, steps=len(t))


def main(Nx=10, Nz=10):
    problem, fault = build_friction_problem(Nx, Nz)

    results = {}
    for label, method in [
        ("velocity_vi_coulomb", problem.as_velocity_vi),
        ("residual_embedded", problem.as_residual_embedded),
    ]:
        try:
            bundle = method()
            results[label] = run_bundle(bundle, fault, label)
        except Exception as e:
            print(f"[{label}] FAILED: {e}")
            results[label] = None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"velocity_vi_coulomb": "C0", "residual_embedded": "C1"}
    markers = {"velocity_vi_coulomb": "o", "residual_embedded": "s"}
    for label, r in results.items():
        if r is None:
            continue
        ax.plot(r["t"], r["mean_v"],
                marker=markers[label], color=colors[label], markersize=4,
                label=f"{label} (wall={r['wall']:.1f}s, steps={r['steps']})")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean slip rate [m/s]")
    ax.set_title(f"2D fault — backend comparison ({Nx}x{Nz})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / f"backends_{Nx}x{Nz}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(Nx=10, Nz=10)
