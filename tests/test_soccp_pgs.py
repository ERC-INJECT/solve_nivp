"""Tests for the block-projected Gauss-Seidel SOCCP solver."""

import numpy as np
import pytest

from solve_nivp.soccp_pgs import (
    soccp_pgs,
    desaxce_shift_factory,
    fremond_shift_factory,
    _local_solve_soccp,
)


def test_frictionless_single_contact_unique_solution():
    # u = 2 p - 1; require 0 <= u perp p >= 0.
    # p=0 -> u=-1<0 (infeasible); p=0.5 -> u=0 (boundary, both feasible).
    p = soccp_pgs(np.array([[2.0]]), np.array([-1.0]),
                  [slice(0, 1)], np.array([0.0]))
    np.testing.assert_allclose(p, [0.5], atol=1.0e-9)


def test_separation_returns_zero_reaction():
    # u = 2 p + 1 has u(0) = 1 > 0, so contact is inactive.
    p = soccp_pgs(np.array([[2.0]]), np.array([1.0]),
                  [slice(0, 1)], np.array([0.0]))
    np.testing.assert_allclose(p, [0.0], atol=1.0e-9)


def test_sticking_single_contact():
    # W = 2 I, b = (-1, 0): u = (2 p_N - 1, 2 p_T).
    # Stick at u=0 gives p = (0.5, 0); ||p_T|| = 0 <= mu p_N = 0.25 ✓.
    mu = 0.5
    p, info = soccp_pgs(2.0 * np.eye(2), np.array([-1.0, 0.0]),
                         [slice(0, 2)], np.array([mu]), return_info=True)
    np.testing.assert_allclose(p, [0.5, 0.0], atol=1.0e-9)
    assert info.regime[0] == "stick"


def test_sliding_single_contact_de_saxce():
    # 2D sliding: tangential drive forces ||p_T|| = mu p_N.
    mu = 0.5
    W = 2.0 * np.eye(2)
    b = np.array([-1.0, -2.0])
    shift = desaxce_shift_factory(np.array([mu]))
    p, info = soccp_pgs(W, b, [slice(0, 2)], np.array([mu]),
                         shift_fn=shift, return_info=True)
    assert info.regime[0] == "slip"
    np.testing.assert_allclose(np.linalg.norm(p[1:]) / p[0], mu,
                                rtol=1.0e-7)


def test_sliding_3d_cone_boundary():
    # 3D sliding: ||p_T|| = mu p_N at cone boundary.
    mu = 0.4
    W = 2.0 * np.eye(3)
    b = np.array([-1.0, -1.5, -0.5])
    shift = desaxce_shift_factory(np.array([mu]))
    p, info = soccp_pgs(W, b, [slice(0, 3)], np.array([mu]),
                         shift_fn=shift, return_info=True)
    assert info.regime[0] == "slip"
    np.testing.assert_allclose(np.linalg.norm(p[1:]) / p[0], mu,
                                rtol=1.0e-7)


def test_two_contacts_coupled_off_block_via_pgs():
    # Two 1D frictionless contacts with off-block coupling.
    # u_1 = W_11 p_1 + W_12 p_2 + b_1
    # u_2 = W_21 p_1 + W_22 p_2 + b_2
    # Both active: W p + b = 0 => p = -W^{-1} b.
    W = np.array([[2.0, -0.5], [-0.5, 2.0]])
    b = np.array([-1.0, -2.0])
    expected = np.linalg.solve(W, -b)
    p = soccp_pgs(W, b, [slice(0, 1), slice(1, 2)],
                  np.array([0.0, 0.0]), max_outer=500)
    np.testing.assert_allclose(p, expected, atol=1.0e-8)


def test_two_contacts_one_separates():
    # Two contacts; w_22 makes contact 2 want p_2 < 0, which is infeasible,
    # so contact 2 separates (p_2 = 0) and contact 1 absorbs the load.
    W = np.array([[2.0, -0.1], [-0.1, 2.0]])
    b = np.array([-1.0, +0.5])  # b_2 > 0 -> u_2 > 0 at p=0 -> separation
    p, info = soccp_pgs(W, b, [slice(0, 1), slice(1, 2)],
                         np.array([0.0, 0.0]), return_info=True)
    np.testing.assert_allclose(p[1], 0.0, atol=1.0e-9)
    assert info.regime[1] == "separation"
    # u_2 at the solution: W_21 p_1 + W_22 * 0 + b_2 = -0.1 * p_1 + 0.5 > 0.
    u2 = W[1, :] @ p + b[1]
    assert u2 > 0.0


def test_de_saxce_shift_reduces_to_identity_when_no_friction():
    mu_vec = np.array([0.0, 0.0])
    shift = desaxce_shift_factory(mu_vec)
    u = np.array([0.5, 0.3])
    np.testing.assert_allclose(shift(u, 0), u)
    np.testing.assert_allclose(shift(u, 1), u)


def test_fremond_shift_collapses_to_de_saxce_at_theta_one_e_zero():
    mu_vec = np.array([0.5])
    fre = fremond_shift_factory(mu_vec, np.array([0.0]),
                                 np.array([-1.0]), theta=1.0)
    des = desaxce_shift_factory(mu_vec)
    u = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(fre(u, 0), des(u, 0))


def test_fremond_shift_average_velocity_at_theta_half():
    # At theta=1/2, e=1: the additional term is (1/2 * 2 - 1) * u_N_old = 0.
    # At theta=1/2, e=0: term is (1/2 - 1) * u_N_old = -0.5 * u_N_old.
    mu_vec = np.array([0.5])
    u_N_old = -2.0
    fre = fremond_shift_factory(mu_vec, np.array([0.0]),
                                 np.array([u_N_old]), theta=0.5)
    u = np.array([0.1, 0.0, 0.0])
    out = fre(u, 0)
    expected = 0.1 + (0.5 - 1.0) * u_N_old
    np.testing.assert_allclose(out[0], expected)


def test_warm_start_reduces_iterations():
    # Same problem solved twice; second run starts from first solution
    # and should converge in a single outer iterate.
    mu = 0.4
    W = 2.0 * np.eye(3)
    b = np.array([-1.0, -1.5, -0.5])
    shift = desaxce_shift_factory(np.array([mu]))
    p_first, info_first = soccp_pgs(W, b, [slice(0, 3)], np.array([mu]),
                                      shift_fn=shift, return_info=True)
    p_warm, info_warm = soccp_pgs(W, b, [slice(0, 3)], np.array([mu]),
                                    shift_fn=shift, p0=p_first,
                                    return_info=True)
    np.testing.assert_allclose(p_warm, p_first, atol=1.0e-9)
    assert info_warm.outer_iters <= info_first.outer_iters


def test_local_solver_exact_residual_at_solution():
    # Verify the Jordan-algebra residual is below tolerance at the solver
    # output, for a 3D coupled SOCCP.
    mu = 0.3
    W = np.array([
        [2.0, 0.1, -0.05],
        [0.1, 2.0, 0.0],
        [-0.05, 0.0, 2.0],
    ])
    b = np.array([-1.2, -0.8, 0.3])
    p_init = np.zeros(3)
    shift = desaxce_shift_factory(np.array([mu]))
    p_local, n_iter, res = _local_solve_soccp(
        W, b, mu, p_init, 0, shift_fn=shift, max_iter=50, tol=1.0e-13,
    )
    assert res < 1.0e-10
    # Sanity: cone admissibility.
    assert p_local[0] >= -1.0e-12
    if p_local[0] > 1.0e-12:
        assert np.linalg.norm(p_local[1:]) <= mu * p_local[0] + 1.0e-9


def test_pgs_does_not_mutate_inputs():
    W = 2.0 * np.eye(2)
    b = np.array([-1.0, -2.0])
    mu_vec = np.array([0.5])
    p0 = np.array([0.4, 0.1])
    W0 = W.copy()
    b0 = b.copy()
    mu0 = mu_vec.copy()
    p00 = p0.copy()
    shift = desaxce_shift_factory(mu_vec)
    soccp_pgs(W, b, [slice(0, 2)], mu_vec, shift_fn=shift, p0=p0)
    np.testing.assert_array_equal(W, W0)
    np.testing.assert_array_equal(b, b0)
    np.testing.assert_array_equal(mu_vec, mu0)
    np.testing.assert_array_equal(p0, p00)
