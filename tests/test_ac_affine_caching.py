import numpy as np

from solve_nivp.alart_curnier_contact import build_alart_curnier_contact


def _build(smooth_rhs_is_affine):
    A = np.eye(2)

    def rhs(t, y):
        return np.array([-y[0] ** 3, y[0]], dtype=float)

    def rhs_jac(t, y):
        return np.array([[-3.0 * y[0] ** 2, 0.0], [1.0, 0.0]], dtype=float)

    def gap_func(y, t):
        return np.array([y[1]], dtype=float)

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[], mu=0.0)]
    B = np.array([[1.0], [0.0]], dtype=float)
    return build_alart_curnier_contact(
        A=A, rhs_smooth=rhs, y0=np.array([0.0, 0.0]), contacts=contacts,
        gap_func=gap_func, B=B, rhs_jac=rhs_jac,
        smooth_rhs_is_affine=smooth_rhs_is_affine,
    )


def _rhs_true(yp):
    return np.array([-yp[0] ** 3, yp[0]])


def test_default_does_not_linearize_nonaffine_rhs():
    """With the default (no affine opt-in) the smooth RHS is evaluated at each
    state; providing rhs_jac must not trigger a stale affine reconstruction."""
    cs = _build(smooth_rhs_is_affine=False)
    a = np.array([2.0, 0.5, 0.0])
    b = np.array([3.0, 0.5, 0.0])
    cs.rhs(0.0, a, a.copy(), 0.01)                 # warm any caches
    out_b = cs.rhs(0.0, b, b.copy(), 0.01)
    np.testing.assert_allclose(out_b[:2], _rhs_true(b[:2]), atol=1.0e-10)


def test_affine_optin_reproduces_cached_linearization():
    """The opt-in fast path snapshots J,b once; for a (deliberately) nonlinear
    RHS that yields a linearization differing from the true RHS — confirming
    the flag, not the mere presence of rhs_jac, controls caching."""
    cs = _build(smooth_rhs_is_affine=True)
    a = np.array([2.0, 0.5, 0.0])
    b = np.array([3.0, 0.5, 0.0])
    cs.rhs(0.0, a, a.copy(), 0.01)                 # snapshot J(a), b
    out_b = cs.rhs(0.0, b, b.copy(), 0.01)
    assert not np.allclose(out_b[:2], _rhs_true(b[:2]), atol=1.0e-6)
