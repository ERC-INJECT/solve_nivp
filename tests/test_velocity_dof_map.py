import numpy as np

from solve_nivp.contact import build_impulse_contact


def test_velocity_dof_map_resolves_extraction_rows_to_physical_dofs():
    """With C_extract, vel_normal_idx/vel_tangential_idx are row indices of
    C_extract, not physical DOFs.  The active-set velocity DOF map must resolve
    them back to the physical DOFs the extraction row touches."""
    n_phys = 4
    A = np.eye(n_phys)

    def rhs(t, y):
        return np.zeros(n_phys, dtype=float)

    C = np.zeros((2, n_phys))
    C[0, 2] = 1.0   # normal extraction row 0 -> physical DOF 2
    C[1, 3] = 1.0   # tangential extraction row 1 -> physical DOF 3

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.5, e=0.0)]
    cs = build_impulse_contact(
        A=A, rhs_smooth=rhs, y0=np.zeros(n_phys), contacts=contacts, C_extract=C)

    vmap = cs.projection._velocity_dof_map
    np.testing.assert_array_equal(np.sort(vmap[0]), [2, 3])
