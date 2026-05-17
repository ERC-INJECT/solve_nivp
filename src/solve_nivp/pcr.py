"""Preconditioned Conjugate Residual (PCR) linear solver.

Solves A x = b where A is accessed only through matrix-vector products.
Unlike CG, PCR's residual norm is monotonically non-increasing, making
it suitable for indefinite and saddle-point systems arising from the
Schur complement of contact complementarity problems.

Reference: Macklin et al. (2019) §8.2, §10.4.
"""

from __future__ import annotations

import numpy as np


def pcr_solve(
    matvec,
    b,
    x0=None,
    maxiter=100,
    tol=1e-10,
    preconditioner=None,
):
    """Solve A x = b using PCR.

    When *preconditioner* is supplied, the algorithm solves the
    left-preconditioned system  M^{-1} A x = M^{-1} b  using CR on the
    effective operator B = M^{-1} A.  Convergence is checked on the true
    (unpreconditioned) residual norm.

    Parameters
    ----------
    matvec : callable
        y = matvec(v) computes y = A v.
    b : ndarray, shape (n,)
    x0 : ndarray or None
        Initial guess; zero if None.
    maxiter : int
    tol : float
        Convergence tolerance on ||r|| / ||b||.
    preconditioner : callable or None
        z = preconditioner(v) applies M^{-1} v. Identity if None.

    Returns
    -------
    x : ndarray
    info : dict
        converged (bool), iterations (int), residual_norm (float),
        residual_history (list[float]).
    """
    n = b.shape[0]
    b_norm = np.linalg.norm(b)

    if b_norm == 0.0:
        return np.zeros(n), dict(
            converged=True, iterations=0,
            residual_norm=0.0, residual_history=[0.0],
        )

    if preconditioner is None:
        Bvec = matvec
        rhs = b
    else:
        Bvec = lambda v: preconditioner(matvec(v))
        rhs = preconditioner(b.copy())

    x = x0.copy() if x0 is not None else np.zeros(n)

    # True residual for convergence check
    r_true = b - matvec(x)
    r_norm = np.linalg.norm(r_true)
    residual_history = [r_norm / b_norm]

    if r_norm / b_norm < tol:
        return x, dict(
            converged=True, iterations=0,
            residual_norm=r_norm / b_norm,
            residual_history=residual_history,
        )

    # Preconditioned residual: s = M^{-1}(b - Ax) = rhs - Bx
    s = rhs - Bvec(x) if preconditioner is not None else r_true.copy()

    p = s.copy()
    Bp = Bvec(p)
    Bs = Bp.copy()
    sBs = np.dot(s, Bs)

    for k in range(maxiter):
        Bp_dot_Bp = np.dot(Bp, Bp)
        if Bp_dot_Bp < 1e-30:
            break

        alpha = sBs / Bp_dot_Bp

        x = x + alpha * p
        s = s - alpha * Bp

        # True residual
        r_true = b - matvec(x)
        r_norm = np.linalg.norm(r_true)
        residual_history.append(r_norm / b_norm)

        if r_norm / b_norm < tol:
            return x, dict(
                converged=True, iterations=k + 1,
                residual_norm=r_norm / b_norm,
                residual_history=residual_history,
            )

        Bs_new = Bvec(s)
        sBs_new = np.dot(s, Bs_new)

        if abs(sBs) < 1e-30:
            break

        beta = sBs_new / sBs

        p = s + beta * p
        Bp = Bs_new + beta * Bp
        sBs = sBs_new

    return x, dict(
        converged=False, iterations=len(residual_history) - 1,
        residual_norm=residual_history[-1],
        residual_history=residual_history,
    )
