"""Backend-agnostic adapter for friction-only 2D fault problems.

Given a pure-friction problem (no Signorini complementarity; sigma_n
prescribed per node), expose it in whatever form each solve_nivp
contact backend needs. Adapter logic lives here so individual benchmark
scripts just pick a backend by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp


class BackendBundle(NamedTuple):
    fun: Callable
    y0: np.ndarray
    A: sp.spmatrix
    projection: object
    projection_opts: dict
    solver_opts: dict
    component_slices: list
    extract_velocity: Callable  # y -> mean slip rate array of physical v


@dataclass
class FrictionOnlyProblem:
    """Minimal description of a pure-friction VI problem.

    State layout of the *base* (unaugmented) system is assumed to be
    ``y_base = [v (N), u (N), S (N)]`` with
      - M dv/dt = f_smooth(y) + reaction
      - du/dt  = v - v_inf
      - dS/dt  = |v|
    """

    rhs_smooth_base: Callable
    rhs_jac_base: Callable
    A_base: sp.spmatrix
    y0_base: np.ndarray
    N: int
    mu_of_slip: Callable          # (slip_vec) -> mu_vec, len N
    sigma_n: np.ndarray           # len N, prescribed normal stress
    component_slices_base: list = field(default_factory=list)

    # --------------------------------------------------------------
    # Backend 1: velocity-level VI with CoulombProjection
    # --------------------------------------------------------------
    def as_velocity_vi(self) -> BackendBundle:
        N = self.N
        sigma = np.ascontiguousarray(self.sigma_n, dtype=np.float64)
        mu_fn = self.mu_of_slip

        con_buf = np.zeros(3 * N, dtype=np.float64)

        def con_force(state, fk=None):
            con_buf.fill(0.0)
            slip = state[2 * N:3 * N]
            mu = mu_fn(slip)
            con_buf[:N] = mu * sigma
            return con_buf.copy()

        projection_opts = {
            "con_force_func": con_force,
            "rhok": np.ones(N, dtype=float),
            "component_slices": self.component_slices_base,
            "constraint_indices": np.arange(N, dtype=np.int32),
            "use_numba": True,
        }

        solver_opts = dict(
            tol=1e-8,
            max_iter=200,
            rhs_jac=self.rhs_jac_base,
            vi_strict_block_lipschitz=False,
            vi_max_block_adjust_iters=5,
            globalization="line_search",
        )

        def extract_v(y_row):
            return y_row[:N]

        return BackendBundle(
            fun=self.rhs_smooth_base,
            y0=self.y0_base.copy(),
            A=self.A_base,
            projection="coulomb",
            projection_opts=projection_opts,
            solver_opts=solver_opts,
            component_slices=self.component_slices_base,
            extract_velocity=extract_v,
        )

    # --------------------------------------------------------------
    # Backend 2: residual-embedded smoothed Coulomb friction
    # --------------------------------------------------------------
    def as_residual_embedded(self, eps_slip: float = 1.0e-3) -> BackendBundle:
        """Embed friction as an augmented algebraic reaction DOF lambda_t.

        State layout: y = [v (N), u (N), S (N), lambda_t (N)], size 4N.
        Descriptor: A = block_diag(MA, I, I, I_eps) with I_eps = eps * I so the
        lambda rows are index-1 algebraic-in-the-limit but numerically stable.
        RHS:
          dv/dt = -(K u + E (v - v_inf)) + lambda_t           (friction acts on momentum)
          du/dt = v - v_inf
          dS/dt = |v|
          eps * d(lambda_t)/dt = lambda_t + mu(S) sigma_n * tanh(v / eps_slip)
                                 (smoothed Coulomb: relaxes toward -mu sigma_n sign(v))
        """
        from solve_nivp.projections import IdentityProjection

        N = self.N
        n_base = 3 * N
        n_aug = 4 * N
        sigma = np.ascontiguousarray(self.sigma_n, dtype=np.float64)
        mu_fn = self.mu_of_slip

        I_N = sp.eye(N, format="csr")
        eps_mass = 1.0e-4
        A_aug = sp.block_diag(
            [self.A_base, eps_mass * I_N], format="csr"
        )

        y0_aug = np.concatenate([self.y0_base, np.zeros(N)])

        rhs_base = self.rhs_smooth_base
        rhs_jac_base = self.rhs_jac_base
        rhs_buf = np.zeros(n_aug, dtype=np.float64)

        def rhs_aug(t, y):
            v = y[:N]
            slip = y[2 * N:3 * N]
            lam = y[3 * N:4 * N]
            base = rhs_base(t, y[:n_base])
            rhs_buf[:n_base] = base
            # Friction reaction couples into momentum row
            rhs_buf[:N] += lam
            mu = mu_fn(slip)
            bound = mu * sigma
            # Smoothed Coulomb residual: drives lambda -> -bound * tanh(v/eps)
            rhs_buf[3 * N:4 * N] = -lam - bound * np.tanh(v / eps_slip)
            return rhs_buf.copy()

        def rhs_jac_aug(t, y, Fk_val=None):
            J = np.zeros((n_aug, n_aug), dtype=np.float64)
            J_base = rhs_jac_base(t, y[:n_base])
            J[:n_base, :n_base] = J_base
            J[:N, 3 * N:4 * N] = np.eye(N)
            v = y[:N]
            slip = y[2 * N:3 * N]
            mu = mu_fn(slip)
            bound = mu * sigma
            arg = np.clip(v / eps_slip, -30.0, 30.0)
            sech2 = 1.0 / np.cosh(arg) ** 2
            J[3 * N + np.arange(N), np.arange(N)] = -bound * sech2 / eps_slip
            J[3 * N + np.arange(N), 3 * N + np.arange(N)] = -1.0
            return J

        component_slices = list(self.component_slices_base) + [
            slice(3 * N, 4 * N)
        ]

        solver_opts = dict(
            tol=1e-8,
            max_iter=200,
            rhs_jac=rhs_jac_aug,
            vi_strict_block_lipschitz=False,
            vi_max_block_adjust_iters=5,
            globalization="line_search",
        )

        def extract_v(y_row):
            return y_row[:N]

        return BackendBundle(
            fun=rhs_aug,
            y0=y0_aug,
            A=A_aug,
            projection=IdentityProjection(component_slices=component_slices),
            projection_opts={},
            solver_opts=solver_opts,
            component_slices=component_slices,
            extract_velocity=extract_v,
        )

    # --------------------------------------------------------------
    # (legacy) impulse-level SOC via build_impulse_contact — kept for reference
    # --------------------------------------------------------------
    def as_impulse_soc(self) -> BackendBundle:
        from solve_nivp.contact import build_impulse_contact

        N = self.N
        sigma = np.ascontiguousarray(self.sigma_n, dtype=np.float64)
        mu_fn = self.mu_of_slip

        # Augment physical state with N dummy normal-velocity DOFs.
        # Layout of the new physical state:
        #   [v (N), u (N), S (N), v_n_dummy (N)]
        n_base = 3 * N
        n_phys = 4 * N

        # Extend A with identity block for the dummy normal DOFs
        I_N = sp.eye(N, format="csr")
        A_phys = sp.block_diag([self.A_base, I_N], format="csr")
        y0_phys = np.concatenate([self.y0_base, np.zeros(N)])

        rhs_base = self.rhs_smooth_base
        rhs_jac_base = self.rhs_jac_base
        rhs_phys_buf = np.zeros(n_phys, dtype=np.float64)

        def rhs_smooth(t, y_phys):
            rhs_phys_buf[:n_base] = rhs_base(t, y_phys[:n_base])
            rhs_phys_buf[n_base:] = 0.0
            return rhs_phys_buf.copy()

        def rhs_jac_phys(t, y_phys, Fk_val=None):
            J = np.zeros((n_phys, n_phys), dtype=np.float64)
            J[:n_base, :n_base] = rhs_jac_base(t, y_phys[:n_base])
            return J

        def gap_func(y_phys, t):
            return np.zeros(N, dtype=np.float64)

        def make_mu_closure(i):
            def _mu(y_phys):
                return float(mu_fn(y_phys[2 * N:3 * N])[i])
            return _mu

        contacts = [
            {
                "vel_normal_idx": n_base + i,
                "vel_tangential_idx": i,
                "mu": make_mu_closure(i),
            }
            for i in range(N)
        ]

        component_slices_phys = list(self.component_slices_base) + [
            slice(n_base, n_phys)
        ]

        cs = build_impulse_contact(
            A=A_phys,
            rhs_smooth=rhs_smooth,
            rhs_jac=rhs_jac_phys,
            y0=y0_phys,
            contacts=contacts,
            gap_func=gap_func,
            component_slices=component_slices_phys,
            gap_tol=1.0,  # always active (closed contact)
            retain_compressive_active=True,
        )

        solver_opts = dict(
            tol=1e-8,
            max_iter=200,
            rhs_jac=cs.rhs_jac,
            vi_strict_block_lipschitz=False,
            vi_max_block_adjust_iters=5,
            globalization="line_search",
        )

        def extract_v(y_row):
            return y_row[:N]

        return BackendBundle(
            fun=cs.rhs,
            y0=cs.y0,
            A=cs.A,
            projection=cs.projection,
            projection_opts={},
            solver_opts=solver_opts,
            component_slices=cs.component_slices,
            extract_velocity=extract_v,
        )
