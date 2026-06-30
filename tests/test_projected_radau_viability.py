import numpy as np

from solve_nivp.projected_radau_contact import build_projected_radau_contact


def _build():
    A = np.eye(3)

    def rhs(t, y):
        return np.zeros(3)

    def jac(t, y):
        J = np.zeros((3, 3))
        J[0, 2] = 10.0
        return J

    cs = build_projected_radau_contact(
        A, rhs, np.zeros(3), [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=np.eye(3), D_extract=np.eye(3),
        B=np.array([[1.0], [0.0], [0.0]]), rhs_jac=jac,
        contact_law="soc_fb_uniform", normal_r=1.0, friction_r=1.0,
        reported_reaction_units="force",
        reaction_state_indices=np.array([2]),
        reaction_state_to_reported_scale=0.5,
        mask_reaction_state_in_smooth_rhs=True,
    )
    return cs.projected_radau_contact


def test_stage_law_gated_for_open_contact_under_velocity_normal():
    m = _build()
    assert getattr(m.contact_law, "expects_velocity_normal", False) is True

    y = np.array([1.0, 0.0, 4.0])     # gap = 1.0 > gap_tol -> open
    assert float(m.gap(y, 0.0)[0]) > m.gap_tol

    F = m._contact_residual(
        y, 0.0, np.array([4.0]), np.array([0.0]), 0.25, endpoint=False)
    # Open contact: Moreau viability deactivates the law, leaving the residual
    # equal to the reported effective reaction (0.5 * 4.0 = 2.0), not the
    # velocity-based contact-law output.
    np.testing.assert_allclose(F, [2.0])

    _Jy, Jr = m._contact_jacobian(
        y, 0.0, np.array([4.0]), np.array([0.0]), 0.25, endpoint=False)
    np.testing.assert_allclose(Jr.toarray(), [[0.5]])
    np.testing.assert_allclose(_Jy.toarray(), np.zeros((1, 3)))
