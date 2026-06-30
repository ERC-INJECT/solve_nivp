import numpy as np
import pytest

from solve_nivp.rattle_contact import (
    RattleBilateralSpec,
    RattleMechanicalSystem,
    RattleContactSpec,
    build_rattle_system,
    RattleSolver,
)


def test_eval_chi_gamma_passes_full_velocity_dimension():
    """eval_chi_gamma evaluates gamma at u = 0; it must pass a zero velocity of
    the correct dimension, not an empty vector."""
    nu = 3
    W_g = np.ones((nu, 1))
    W_gamma = np.array([[1.0], [0.0], [0.0]])
    chi = np.array([0.7])

    def gamma(t, q, u):
        return W_gamma.T @ np.asarray(u, dtype=float) + chi

    spec = RattleBilateralSpec(
        g=lambda t, q: np.zeros(1), W_g=W_g, n_g=1,
        gamma=gamma, W_gamma=W_gamma, n_gamma=1)

    out = spec.eval_chi_gamma(0.0, np.zeros(nu))
    np.testing.assert_allclose(out, chi)


def test_stage1_jacobian_rejects_configuration_dependent_B():
    """Configuration-dependent B (dB_dq) is not implemented in the Stage-1
    Jacobian; it must raise rather than silently drop the term."""
    mech = RattleMechanicalSystem(
        nq=1, nu=1, q0=np.array([0.0]), u0=np.array([0.0]),
        M=np.array([[2.0]]), h_force=lambda t, q, u: np.zeros(1))
    contact = RattleContactSpec(
        g_N=lambda t, q: float(q[0]), W_N=np.array([1.0]),
        gamma_F=lambda t, q, u: np.zeros(1), W_F=np.array([[0.0]]),
        mu=0.0, n_F=1)
    system = build_rattle_system(mech, contacts=[contact], prox_alpha=0.5)
    solver = RattleSolver(system)
    solver.mech.dB_dq = lambda t, q: np.zeros((1, 1))

    x = np.zeros(solver._nx1)
    with pytest.raises(NotImplementedError):
        solver._stage1_jacobian(
            x, t_n=0.0, h=0.01, q_n=mech.q0, u_n=mech.u0,
            B_n=np.eye(1), M_n=mech.M)
