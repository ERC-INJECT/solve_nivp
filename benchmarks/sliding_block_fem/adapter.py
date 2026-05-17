"""Adapter: wrap BouncingBlockProblem into solver backends."""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp

from problem import BouncingBlockProblem
from solve_nivp.rattle_contact import (
    RattleMechanicalSystem,
    RattleContactSpec,
    RattleContactSystem,
    build_rattle_system,
)


class RattleBundle(NamedTuple):
    system: RattleContactSystem
    label: str


class SNBundle(NamedTuple):
    fun: Callable
    y0: np.ndarray
    A: object
    projection: object
    projection_opts: dict
    solver_opts: dict
    component_slices: list
    n_phys: int
    label: str
    integrator_opts: dict


class NCPBundle(NamedTuple):
    fun: Callable
    y0: np.ndarray
    A: object
    projection: object
    projection_opts: dict
    solver_opts: dict
    component_slices: list
    n_phys: int
    label: str
    integrator_opts: dict


def as_ncp(p: BouncingBlockProblem,
           normal_r="auto",
           friction_r="auto",
           gap_tol: float = 0.0,
           ncp_type: str = "fischer_burmeister",
           inactive_handling: str = "ncp",
           rate_form: bool = False) -> NCPBundle:
    from solve_nivp.ncp_contact import build_ncp_contact

    cs = build_ncp_contact(
        A=p.A,
        rhs_smooth=p.rhs_smooth,
        rhs_jac=p.rhs_jac,
        y0=p.y0,
        contacts=p.contacts,
        gap_func=p.gap_func,
        component_slices=p.component_slices,
        gap_tol=gap_tol,
        normal_r=normal_r,
        friction_r=friction_r,
        ncp_type=ncp_type,
        inactive_handling=inactive_handling,
        rate_form=rate_form,
    )
    solver_opts = dict(
        tol=1.0e-7,
        max_iter=500,
        rhs_jac=cs.rhs_jac,
        globalization="damped",
        damped_step_fraction=0.75,
        diagonal_regularization=1.0e-6,
        cold_start_slices=[cs.component_slices[0], cs.component_slices[-1]],
        modified_newton_identity=False,
    )
    return NCPBundle(
        fun=cs.rhs,
        y0=cs.y0,
        A=cs.A,
        projection=cs.projection,
        projection_opts={},
        solver_opts=solver_opts,
        component_slices=cs.component_slices,
        n_phys=p.n_phys,
        label="ncp",
        integrator_opts=dict(getattr(cs, "integrator_opts", {}) or {}),
    )


def as_rattle(p: BouncingBlockProblem,
              prox_alpha: float = 0.5,
              gap_tol: float = 0.0) -> RattleBundle:
    nq = p.n_dof
    nu = p.n_dof

    q0 = np.zeros(nq)
    u0 = np.zeros(nu)

    K = p.K
    f_grav = -(K @ q0)  # placeholder; gravity vector is computed below

    # Pull gravity vector out of rhs_smooth at y=[0,0]: the momentum row
    # contains -K q + f_grav, so at q=0,u=0 it's exactly f_grav.
    y_zero = np.zeros(p.n_phys)
    f_rhs = p.rhs_smooth(0.0, y_zero)
    grav_vec = f_rhs[:nq].copy()

    def h_force(t, q, u):
        return -(K @ q) + grav_vec

    def dh_dq(t, q, u):
        return -K

    def dh_du(t, q, u):
        return sp.csr_matrix((nu, nu))

    mech = RattleMechanicalSystem(
        nq=nq, nu=nu, q0=q0, u0=u0,
        M=p.M, h_force=h_force,
        dh_dq=dh_dq, dh_du=dh_du,
    )

    uy_idx = 2 * p.bottom_nodes + 1
    ux_idx = 2 * p.bottom_nodes
    bottom_y_ref = p.bottom_node_y_ref

    contact_specs = []
    for k in range(len(p.bottom_nodes)):
        i_n = int(uy_idx[k])
        i_t = int(ux_idx[k])
        y_ref = float(bottom_y_ref[k])

        W_N = np.zeros(nu)
        W_N[i_n] = 1.0
        W_F = np.zeros((nu, 1))
        W_F[i_t, 0] = 1.0

        def _gap(t, q, i_n=i_n, y_ref=y_ref):
            return y_ref + q[i_n]

        def _slip(t, q, u, i_t=i_t):
            return np.array([u[i_t]])

        contact_specs.append(
            RattleContactSpec(
                g_N=_gap,
                W_N=W_N,
                gamma_F=_slip,
                W_F=W_F,
                mu=float(p.mu),
                e=1.0,
                n_F=1,
            )
        )

    system = build_rattle_system(
        mech,
        contacts=contact_specs,
        prox_alpha=prox_alpha,
        gap_tol=gap_tol,
    )
    return RattleBundle(system=system, label="rattle")


def as_desaxce(p: BouncingBlockProblem,
               contact_rho="auto",
               gap_tol: float = 0.0,
               reaction_units: str = "impulse") -> SNBundle:
    from solve_nivp.desaxce_contact import build_dynamic_desaxce_residual_contact

    cs = build_dynamic_desaxce_residual_contact(
        A=p.A,
        rhs_smooth=p.rhs_smooth,
        rhs_jac=p.rhs_jac,
        y0=p.y0,
        contacts=p.contacts,
        gap_func=p.gap_func,
        component_slices=p.component_slices,
        gap_tol=gap_tol,
        contact_rho=contact_rho,
        reaction_units=reaction_units,
        inactive_handling="hard_zero",
    )
    solver_opts = dict(
        tol=1.0e-7,
        max_iter=500,
        rhs_jac=cs.rhs_jac,
        globalization="damped",
        damped_step_fraction=0.75,
        diagonal_regularization=1.0e-6,
    )
    return SNBundle(
        fun=cs.rhs,
        y0=cs.y0,
        A=cs.A,
        projection=cs.projection,
        projection_opts={},
        solver_opts=solver_opts,
        component_slices=cs.component_slices,
        n_phys=p.n_phys,
        label="desaxce",
        integrator_opts=dict(getattr(cs, "integrator_opts", {}) or {}),
    )
