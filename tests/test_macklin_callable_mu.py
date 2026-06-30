import numpy as np
import pytest

from solve_nivp.macklin_contact import build_macklin_contact_blocked


def test_callable_mu_rejected():
    """The Macklin backend stores a scalar mu per contact; a callable mu must
    be rejected explicitly rather than silently treated as frictionless."""
    A = np.diag([1.0, 1.0, 1.0, 1.0])

    def rhs(t, y):
        return np.concatenate([np.array([0.0, -9.81]), y[:2]])

    def gap_func(y, t):
        return np.array([y[3]])

    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0],
                     mu=lambda slip: 0.5)]
    y0 = np.array([0.0, 0.0, 0.0, 0.5])

    with pytest.raises((TypeError, ValueError)):
        build_macklin_contact_blocked(
            A=A, rhs_smooth=rhs, y0=y0, contacts=contacts, gap_func=gap_func)
