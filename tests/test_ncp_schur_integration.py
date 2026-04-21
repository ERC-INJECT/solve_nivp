"""End-to-end tests for NCP + Schur complement Newton solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solve_nivp.ncp_contact import build_ncp_contact_blocked
from solve_nivp.block_system import SchurComplementSolver


def _spring_impact_setup():
    """1-DOF spring-loaded impact: mass on spring above a floor."""
    mass = 1.0
    stiffness = 60.0
    damping = 0.1
    anchor = 0.4
    gravity = 9.81
    A = np.diag([mass, 1.0])

    def rhs(t, y):
        v, q = y
        return np.array(
            [-stiffness * (q - anchor) - damping * v - mass * gravity, v],
        )

    def rhs_jac(t, y):
        return np.array([[-damping, -stiffness], [1.0, 0.0]])

    def gap_func(y, t):
        return np.array([y[1]])

    y0 = np.array([-2.0, 0.25])
    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[], mu=0.0)]
    return A, rhs, rhs_jac, y0, contacts, gap_func


def test_block_assembly_returns_correct_shapes():
    A, rhs, rhs_jac, y0, contacts, gap_func = _spring_impact_setup()
    bs = build_ncp_contact_blocked(
        A=A, rhs_smooth=rhs, y0=y0, contacts=contacts,
        gap_func=gap_func, rhs_jac=rhs_jac,
    )

    assert bs.n_phys == 2
    assert bs.n_react == 1

    y_aug = np.zeros(bs.n_phys + bs.n_react)
    y_aug[:bs.n_phys] = y0
    blocks = bs.assemble_blocks(y_aug, t=0.01, h=0.01, y_prev=y_aug)

    assert blocks["H"].shape == (2, 2)
    assert blocks["J"].shape == (1, 2)
    assert blocks["C"].shape == (1, 1)
    assert blocks["g"].shape == (2,)
    assert blocks["h_c"].shape == (1,)


def test_precond_diag_positive():
    A, rhs, rhs_jac, y0, contacts, gap_func = _spring_impact_setup()
    bs = build_ncp_contact_blocked(
        A=A, rhs_smooth=rhs, y0=y0, contacts=contacts,
        gap_func=gap_func, rhs_jac=rhs_jac,
    )
    y_aug = np.zeros(bs.n_phys + bs.n_react)
    y_aug[:bs.n_phys] = y0
    blocks = bs.assemble_blocks(y_aug, t=0.01, h=0.01, y_prev=y_aug)

    assert np.all(blocks["precond_diag"] > 0.0)


def test_schur_solve_matches_direct_spring_impact():
    """Schur complement Newton matches the monolithic solver on spring impact."""
    import solve_nivp
    from solve_nivp.ncp_contact import build_ncp_contact

    A, rhs, rhs_jac, y0, contacts, gap_func = _spring_impact_setup()

    bs = build_ncp_contact_blocked(
        A=A, rhs_smooth=rhs, y0=y0, contacts=contacts,
        gap_func=gap_func, rhs_jac=rhs_jac,
        reaction_units="force",
    )
    n_p, n_r = bs.n_phys, bs.n_react
    y_aug = np.zeros(n_p + n_r)
    y_aug[:n_p] = y0

    h = 0.001
    t = h
    solver = SchurComplementSolver(
        maxiter=20, tol=1e-10, pcr_maxiter=50, pcr_tol=1e-12,
    )
    y_new, err, converged, iters = solver.solve(bs, y_aug, t, h, y_aug)
    assert converged, f"Schur Newton did not converge: err={err}, iters={iters}"

    # Monolithic reference via solve_nivp
    cs_mono = build_ncp_contact(
        A=A, rhs_smooth=rhs, y0=y0, contacts=contacts,
        gap_func=gap_func, reaction_units="force",
    )
    t_span = (0, h)
    t_mono, y_mono, *_ = solve_nivp.solve_nivp(
        fun=cs_mono.rhs, t_span=t_span, y0=cs_mono.y0,
        A=cs_mono.A, projection=cs_mono.projection,
        component_slices=cs_mono.component_slices,
        method="backward_euler", adaptive=False, h0=h,
        solver="semismooth_newton",
        integrator_opts=cs_mono.integrator_opts,
        solver_opts={
            "tol": 1e-12, "max_iter": 50,
            "globalization": "linesearch", "linear_solver": "splu",
            "rhs_jac": cs_mono.rhs_jac,
        },
    )
    y_mono_final = y_mono[-1]

    assert_allclose(
        y_new[:n_p], y_mono_final[:n_p], atol=1e-6,
        err_msg="Schur and monolithic physical states diverge",
    )
