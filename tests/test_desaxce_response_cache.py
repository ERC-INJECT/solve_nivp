import numpy as np

from solve_nivp.desaxce_contact import build_dynamic_desaxce_contact


def _build():
    """Frictionless normal contact with a state-dependent smooth Jacobian
    (cubic restoring force), so the physical response matrix depends on state."""
    A = np.eye(2)

    def rhs(t, y):
        v, q = y
        return np.array([-q ** 3, v], dtype=float)

    def rhs_jac(t, y, Fk=None):
        v, q = y
        J = np.zeros((2, 2), dtype=float)
        J[0, 1] = -3.0 * q ** 2
        J[1, 0] = 1.0
        return J

    def gap_func(y, t):
        return np.array([y[1]], dtype=float)

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[], mu=0.0, e=0.0)]
    B = np.array([[1.0], [0.0]], dtype=float)
    return build_dynamic_desaxce_contact(
        A=A, rhs_smooth=rhs, rhs_jac=rhs_jac,
        y0=np.array([0.0, 1.0]), contacts=contacts, gap_func=gap_func, B=B,
    )


def _G_state_at(projection, state, h):
    active = projection._active_blocks(0.0, state, state)
    local_model = projection.local_model_builder(
        t=0.0, current_state=state, candidate=state,
        active_blocks=active, prev_state=state.copy(), step_size=h)
    return np.asarray(local_model["G_state"], dtype=float)


def test_response_matrix_not_stale_for_nonaffine_rhs():
    state_a = np.array([-1.0, -0.5])   # penetrating, closing
    state_b = np.array([-1.0, -2.0])   # different q -> different Jacobian
    h = 0.1

    fresh = _build()
    G_b_fresh = _G_state_at(fresh.projection, state_b, h)

    shared = _build()
    _G_state_at(shared.projection, state_a, h)        # populate cache at A
    G_b_shared = _G_state_at(shared.projection, state_b, h)

    # sanity: the Delassus block really is state-dependent here
    assert not np.allclose(_G_state_at(_build().projection, state_a, h),
                           G_b_fresh, atol=1.0e-6)

    np.testing.assert_allclose(G_b_shared, G_b_fresh, atol=1.0e-10)
