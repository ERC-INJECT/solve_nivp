"""Regressions for the ``t_eval`` output-time feature on solve_nivp."""

import numpy as np
import pytest

import solve_nivp


def _decay_rhs(t, y):
    return -y


def _harmonic_rhs(t, y):
    return np.array([y[1], -y[0]])


def test_t_eval_lands_exactly_on_requested_points_adaptive():
    """Output times must be element-wise equal to the requested t_eval."""
    t_eval = np.array([0.0, 0.1, 0.4, 0.9, 1.5, 2.0])
    t, y, h, _, _ = solve_nivp.solve_nivp(
        fun=_decay_rhs, t_span=(0.0, 2.0), y0=np.array([1.0]),
        method='backward_euler', solver='semismooth_newton',
        adaptive=True, h0=0.05, atol=1e-10, rtol=1e-8,
        t_eval=t_eval,
    )
    np.testing.assert_array_equal(t, t_eval)
    assert y.shape == (t_eval.size, 1)


def test_t_eval_accuracy_matches_exact_solution():
    """Adaptive solve at t_eval points must converge to the exact solution."""
    t_eval = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    _, y, *_ = solve_nivp.solve_nivp(
        fun=_decay_rhs, t_span=(0.0, 2.0), y0=np.array([1.0]),
        method='backward_euler', solver='semismooth_newton',
        adaptive=True, h0=0.01, atol=1.0e-8, rtol=1.0e-6,
        t_eval=t_eval,
    )
    np.testing.assert_allclose(y.flatten(), np.exp(-t_eval), rtol=5.0e-3)


def test_t_eval_works_with_radau_iia():
    """t_eval must work with the higher-order RadauIIA integrator."""
    t_eval = np.linspace(0.0, np.pi, 7)
    y0 = np.array([1.0, 0.0])
    _, y, *_ = solve_nivp.solve_nivp(
        fun=_harmonic_rhs, t_span=(0.0, np.pi), y0=y0,
        method='radau_iia', solver='semismooth_newton',
        integrator_opts=dict(stages=2),
        adaptive=True, h0=0.05, atol=1e-10, rtol=1e-8,
        t_eval=t_eval,
    )
    np.testing.assert_allclose(y[:, 0], np.cos(t_eval), atol=1.0e-4)
    np.testing.assert_allclose(y[:, 1], -np.sin(t_eval), atol=1.0e-4)


def test_t_eval_works_in_fixed_step_mode():
    """Fixed-step integration also clips to t_eval points."""
    t_eval = np.array([0.0, 0.3, 0.7, 1.0])
    t, y, *_ = solve_nivp.solve_nivp(
        fun=_decay_rhs, t_span=(0.0, 1.0), y0=np.array([1.0]),
        method='backward_euler', solver='semismooth_newton',
        adaptive=False, h0=0.01,
        t_eval=t_eval,
    )
    np.testing.assert_array_equal(t, t_eval)


def test_t_eval_skipping_t0_returns_only_listed_points():
    """When t_eval omits t0, the initial state is not included in output."""
    t_eval = np.array([0.5, 1.0, 1.5])
    t, y, *_ = solve_nivp.solve_nivp(
        fun=_decay_rhs, t_span=(0.0, 1.5), y0=np.array([1.0]),
        method='backward_euler', solver='semismooth_newton',
        adaptive=True, h0=0.05,
        t_eval=t_eval,
    )
    np.testing.assert_array_equal(t, t_eval)
    assert y.shape == (3, 1)


def test_t_eval_validates_monotone():
    with pytest.raises(ValueError, match='strictly increasing'):
        solve_nivp.solve_nivp(
            fun=_decay_rhs, t_span=(0.0, 1.0), y0=np.array([1.0]),
            t_eval=np.array([0.0, 0.5, 0.4, 1.0]),
        )


def test_t_eval_validates_range():
    with pytest.raises(ValueError, match='out of range'):
        solve_nivp.solve_nivp(
            fun=_decay_rhs, t_span=(0.0, 1.0), y0=np.array([1.0]),
            t_eval=np.array([0.0, 0.5, 1.5]),
        )
