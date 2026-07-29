"""Cold-restart fallback for the contact semismooth Newton solve.

When the warm-started attempt exhausts ``contact_ssn_max_iter`` without
reaching tol, the solver retries once from ``p = 0`` before raising.
"""
import numpy as np
import pytest

from solve_nivp.moreau_jean_fremond import MoreauJeanFremondStepper
from solve_nivp.nonlinear_solvers import PETSC_AVAILABLE


def _one_contact_stepper(**opts):
    """Minimal 1-contact (normal+tangential) stepper for direct SSN calls."""
    return MoreauJeanFremondStepper(
        M=np.eye(2), K=np.zeros((2, 2)), C=np.zeros((2, 2)),
        block_slices=[slice(0, 2)], e_N_vec=np.zeros(1),
        H_callable=np.eye(2), F_callable=lambda t: np.zeros(2),
        theta=0.5,
        aux_law=lambda aux, u, sl, h, p: dict(aux),
        contact_solver="petsc_ssn",
        **opts,
    )


# Separated contact: at p = 0 the relative velocity u = b has u_N > 0, so the
# exact solution is p* = 0.  The warm start is deep in the cone interior and
# far from p*; with a tight iteration cap the warm-started Newton cannot reach
# tol, while a cold start from p = 0 is already converged.
_W = np.diag([10.0, 10.0])
_B = np.array([2.0, 0.3])
_MU = np.array([0.5])
_P0_BAD = np.array([50.0, 10.0])


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_ssn_cold_restart_recovers_from_bad_warm_start():
    stepper = _one_contact_stepper(contact_ssn_max_iter=1)
    p, info, diag = stepper._solve_contact_petsc_ssn(
        _W, _B, _MU, None, _P0_BAD.copy(),
    )
    assert diag["contact_ssn_converged"]
    np.testing.assert_allclose(p, np.zeros(2), atol=1.0e-8)
    assert info.converged


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_ssn_cold_restart_disabled_keeps_hard_failure():
    stepper = _one_contact_stepper(
        contact_ssn_max_iter=1, contact_ssn_cold_restart=False,
    )
    with pytest.raises(RuntimeError, match="SSN failed to converge"):
        stepper._solve_contact_petsc_ssn(_W, _B, _MU, None, _P0_BAD.copy())


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_ssn_zero_warm_start_failure_still_raises():
    # a zero warm start would make the cold restart repeat the identical
    # attempt, so it is skipped and the failure surfaced as before
    W = np.array([[10.0, 9.9], [9.9, 10.0]])
    b = np.array([-5.0, 40.0])
    stepper = _one_contact_stepper(contact_ssn_max_iter=1)
    with pytest.raises(RuntimeError, match="SSN failed to converge"):
        stepper._solve_contact_petsc_ssn(W, b, _MU, None, np.zeros(2))
