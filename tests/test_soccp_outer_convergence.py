import numpy as np

from solve_nivp.soccp_pgs import soccp_pgs


def test_outer_convergence_requires_local_residual():
    """The outer PGS must not declare convergence on a small parameter update
    alone when the per-block local SOCCP residual is still large."""
    W = np.array([[2.0, 0.3], [0.3, 1.5]])
    b = np.array([-1.0, 0.2])
    block_slices = [slice(0, 2)]
    mu_vec = np.array([0.5])

    # max_inner=0: local solves make no progress, so p stays at p0 (zero
    # parameter update) while the local residual remains large.
    p, info = soccp_pgs(
        W, b, block_slices, mu_vec, p0=np.zeros(2),
        max_inner=0, return_info=True)

    assert info.local_residual_max > 1.0e-6
    assert info.converged is False
