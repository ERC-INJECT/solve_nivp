import numpy as np
import pytest

from solve_nivp import solve_nivp


def _rhs(t, y):
    return -y


def test_theta_method_accepts_theta_in_integrator_opts():
    """A user-supplied ``theta`` must not collide with the hard-coded default."""
    t, y, _h, _fk, _info = solve_nivp(
        _rhs, (0.0, 0.1), np.array([1.0]),
        method='theta', projection='identity', solver='VI',
        integrator_opts={'theta': 0.7},
    )
    assert np.all(np.isfinite(y))


def test_embedded_betr_tolerates_shared_integrator_opts():
    """``embedded_betr`` accepts the same BackwardEuler-family options the other
    integrators do, so a uniform integrator_opts dict does not crash it."""
    t, y, _h, _fk, _info = solve_nivp(
        _rhs, (0.0, 0.1), np.array([1.0]),
        method='embedded_betr', projection='identity', solver='VI',
        integrator_opts={'pass_prev_state': True},
    )
    assert np.all(np.isfinite(y))


def test_embedded_betr_forwards_unknown_integrator_opts():
    """A genuinely unknown option still surfaces (opts are forwarded, not
    silently dropped), matching the other methods."""
    with pytest.raises(TypeError):
        solve_nivp(
            _rhs, (0.0, 0.1), np.array([1.0]),
            method='embedded_betr', projection='identity', solver='VI',
            integrator_opts={'definitely_not_a_real_option': 1},
        )


def test_reserved_solver_opts_key_is_rejected():
    """Reserved constructor kwargs passed through solver_opts must raise a
    clear ValueError rather than a duplicate-keyword TypeError."""
    with pytest.raises(ValueError):
        solve_nivp(
            _rhs, (0.0, 0.1), np.array([1.0]),
            method='backward_euler', projection='identity', solver='VI',
            solver_opts={'nl_rtol': 0.0},
        )
