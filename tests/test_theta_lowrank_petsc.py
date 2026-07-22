import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.moreau_jean_fremond import (
    _ThetaFactorization,
    DescriptorMoreauJeanFremondStepper,
    PETSC_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py required")

MUMPS_OPTS = {"ksp_type": "preonly", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}


def _random_system(n=60, k=3, seed=7):
    rng = np.random.default_rng(seed)
    op_sparse = sp.random(n, n, density=0.08, random_state=seed, format="csr")
    op_sparse = op_sparse + sp.eye(n) * (n * 0.5)   # well-conditioned
    U = rng.standard_normal((n, k))
    V = rng.standard_normal((n, k))
    coef = 0.37
    op_full = op_sparse.toarray() + coef * (U @ V.T)
    return op_sparse, U, V, coef, op_full


def test_petsc_lowrank_single_rhs_matches_dense():
    op_sparse, U, V, coef, op_full = _random_system()
    fac = _ThetaFactorization(op_sparse, "petsc", MUMPS_OPTS,
                              lowrank=(U, V, coef))
    rhs = np.arange(1.0, op_sparse.shape[0] + 1.0)
    x = fac.solve(rhs)
    x_ref = np.linalg.solve(op_full, rhs)
    assert np.allclose(x, x_ref, rtol=1e-11, atol=1e-11)
    fac.destroy()


def test_petsc_lowrank_multi_rhs_matches_dense():
    op_sparse, U, V, coef, op_full = _random_system()
    RHS = np.random.default_rng(1).standard_normal((op_sparse.shape[0], 5))
    fac = _ThetaFactorization(op_sparse, "petsc", MUMPS_OPTS,
                              lowrank=(U, V, coef))
    X = fac.solve_multi(RHS)
    X_ref = np.linalg.solve(op_full, RHS)
    assert np.allclose(X, X_ref, rtol=1e-11, atol=1e-11)
    fac.destroy()


def test_scipy_lowrank_still_works():
    op_sparse, U, V, coef, op_full = _random_system()
    fac = _ThetaFactorization(op_sparse, "scipy", None, lowrank=(U, V, coef))
    rhs = np.ones(op_sparse.shape[0])
    assert np.allclose(fac.solve(rhs), np.linalg.solve(op_full, rhs),
                       rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# Stepper-level contract: a DescriptorMoreauJeanFremondStepper driven
# with (J_sparse, theta_lowrank_jac=(U, V)) must integrate to the SAME
# trajectory as one driven with J_folded = J_sparse - U V^T baked into the
# sparse Jacobian -- both on the MUMPS backend, contact-free.
#
# This pins two facts (stepper contract, moreau_jean_fremond.py:1596-1602,
# 1773-1781):
#   1. rhs_callable returns the FULL physics (including the -U V^T y term) in
#      BOTH configurations;
#   2. the (U, V) sign convention is  J_true = rhs_jac - U V^T  (no flip):
#      op = op_sparse + h U V^T restores  (1/theta)A - h J_folded,  and the
#      affine RHS is corrected by  + h U V^T y  so both linearize about the
#      same operator.
# ---------------------------------------------------------------------------
def _lowrank_trajectory_stepper(J_use, lowrank, *, n, theta, J_folded, b):
    """Contact-free descriptor stepper, MUMPS backend.

    ``rhs_callable`` is the FULL physics ``f(y) = J_folded @ y + b`` in BOTH
    the folded reference and the low-rank split -- the only difference is which
    Jacobian ``rhs_jac_callable`` advertises (J_folded vs J_sparse) and whether
    the residual -U V^T is carried as an explicit ``theta_lowrank_jac`` term.
    """
    A = sp.eye(n, format="csr")
    D_extract = np.zeros((0, n))          # no contact velocity rows
    B = np.zeros((n, 0))                  # no reaction columns

    def rhs(t, y):
        return np.asarray(J_folded @ np.asarray(y).ravel()).ravel() + b

    def jac(t, y):
        return J_use

    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=jac,
        D_extract=D_extract, B=B, contacts=[],
        theta=theta,
        theta_linear_solver="petsc", theta_petsc_options=MUMPS_OPTS,
        theta_lowrank_jac=lowrank,
    )


def test_stepper_lowrank_trajectory_matches_folded_jacobian():
    """theta step with (J_sparse, theta_lowrank_jac=(U, V)) == theta step with
    J_folded = J_sparse - U V^T baked into the sparse Jacobian, MUMPS backend."""
    rng = np.random.default_rng(3)
    n, k, h, theta = 40, 2, 0.05, 0.5
    J_sparse = (sp.random(n, n, density=0.1, random_state=3, format="csr")
                - sp.eye(n) * 2.0).tocsr()
    U = rng.standard_normal((n, k)) * 0.3
    V = rng.standard_normal((n, k)) * 0.3
    J_folded = (J_sparse - sp.csr_matrix(U @ V.T)).tocsr()
    b = rng.standard_normal(n)

    s_ref = _lowrank_trajectory_stepper(
        J_folded, None, n=n, theta=theta, J_folded=J_folded, b=b)
    s_lr = _lowrank_trajectory_stepper(
        J_sparse, (U, V), n=n, theta=theta, J_folded=J_folded, b=b)

    y_ref = np.ones(n)
    y_lr = np.ones(n)
    aux_ref, aux_lr = {}, {}
    for kstep in range(10):
        y_ref, aux_ref, _ = s_ref.step(kstep * h, y_ref, aux_ref, h)
        y_lr, aux_lr, _ = s_lr.step(kstep * h, y_lr, aux_lr, h)
    assert np.allclose(y_lr, y_ref, rtol=1e-11, atol=1e-12)
