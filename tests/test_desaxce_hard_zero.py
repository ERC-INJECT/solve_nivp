import numpy as np

from solve_nivp.desaxce_contact import build_dynamic_desaxce_residual_contact


def test_hard_zero_open_contact_enforces_zero_reaction():
    """An open contact (gap > tol) in hard_zero mode must make r = 0 the root;
    a prestress offset must not shift it to r = -offset."""
    A = np.eye(2)

    def rhs(t, y):
        return np.array([0.0, y[0]], dtype=float)

    def rhs_jac(t, y, Fk=None):
        J = np.zeros((2, 2), dtype=float)
        J[1, 0] = 1.0
        return J

    def gap_func(y, t):
        return np.array([y[1]], dtype=float)

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[], mu=0.0)]
    B = np.array([[1.0], [0.0]], dtype=float)

    cs = build_dynamic_desaxce_residual_contact(
        A=A, rhs_smooth=rhs, rhs_jac=rhs_jac, y0=np.array([0.0, 0.0]),
        contacts=contacts, gap_func=gap_func, B=B,
        inactive_handling="hard_zero",
        get_s0=lambda y: np.array([4000.0]),
        reaction_units="force",
    )

    y = cs.y0.copy()
    y[:2] = [0.0, 0.5]    # open contact: gap = 0.5 > tol
    y[2:] = [0.0]         # zero reaction
    res = cs.rhs(0.0, y, y.copy(), None, 0.01)

    assert abs(res[2]) < 1.0e-9
