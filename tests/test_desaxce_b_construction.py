import numpy as np

from solve_nivp.desaxce_contact import build_dynamic_desaxce_contact


def _friction_setup():
    A = np.eye(2)

    def rhs(t, y):
        return np.zeros(2, dtype=float)

    def rhs_jac(t, y, Fk=None):
        return np.zeros((2, 2), dtype=float)

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.5, e=0.0)]
    # Normal-only gap map (1 row); full velocity map (normal + tangential).
    gap_extract = np.array([[1.0, 0.0]])
    vel_extract = np.array([[1.0, 0.0], [0.0, 1.0]])
    return A, rhs, rhs_jac, contacts, gap_extract, vel_extract


def _build(B):
    A, rhs, rhs_jac, contacts, gap_extract, vel_extract = _friction_setup()
    return build_dynamic_desaxce_contact(
        A=A, rhs_smooth=rhs, rhs_jac=rhs_jac, y0=np.zeros(2),
        contacts=contacts, gap_extract=gap_extract, vel_extract=vel_extract, B=B,
    )


def test_friction_B_built_from_vel_extract_not_gap_extract():
    reaction_rows = [0, 1]   # v_n=0, v_t=[1]
    _A, _r, _j, _c, _g, vel_extract = _friction_setup()
    B_explicit = vel_extract[reaction_rows, :].T

    cs_auto = _build(None)               # builder must use vel_extract for B
    cs_explicit = _build(B_explicit)

    state = np.array([-1.0, 0.5])        # closing normal, sliding tangential
    kw = dict(t=0.0, prev_state=state.copy(), step_size=0.1)
    r_auto = cs_auto.projection.reaction_from_state(state, **kw)
    r_explicit = cs_explicit.projection.reaction_from_state(state, **kw)

    np.testing.assert_allclose(r_auto, r_explicit, atol=1.0e-12)
