"""Block-structured Newton solver using Schur-complement reduction.

Implements the compliance-formulation Newton method from Macklin et al.
(2019) §6.  The 2×2 block saddle-point system

    [H    -J^T] [Δu]   [g]
    [J      C ] [Δλ] = [h]

is reduced via Schur complement to the n_react × n_react system

    [J H⁻¹ J^T + C] Δλ = h - J H⁻¹ g

solved with PCR, then back-substituted: Δu = H⁻¹ (g + J^T Δλ).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .pcr import pcr_solve


@runtime_checkable
class BlockStructuredSystem(Protocol):
    """Protocol for systems that expose block decomposition."""

    n_phys: int
    n_react: int

    def assemble_blocks(
        self,
        y: np.ndarray,
        t: float,
        h: float,
        y_prev: np.ndarray,
    ) -> Dict[str, Any]:
        """Return the block components for the saddle-point system.

        Returns
        -------
        dict with keys:
            H : ndarray or sparse, (n_phys, n_phys)
                Mass matrix minus geometric stiffness: M - h² K.
            J : ndarray or sparse, (n_react, n_phys)
                Constraint Jacobian mapping velocities to constraint space.
            C : ndarray or sparse, (n_react, n_react)
                NCP compliance block (∂φ/∂λ derivatives).
            g : ndarray, (n_phys,)
                Momentum residual.
            h_c : ndarray, (n_react,)
                Contact / constraint residual.
            precond_diag : ndarray, (n_react,) or None
                Diagonal preconditioner r_i = [J H⁻¹ J^T]_{ii}.
        """
        ...


class SchurComplementSolver:
    """Newton solver using Schur-complement reduction with PCR.

    Parameters
    ----------
    maxiter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on the full residual norm.
    pcr_maxiter : int
        Maximum PCR iterations for the Schur system.
    pcr_tol : float
        PCR convergence tolerance.
    damped_step_fraction : float
        Macklin §8.1 damped Newton step t ∈ (0, 1].
    diagonal_regularization : float
        Macklin §8.4 ε added to Schur diagonal.
    use_preconditioner : bool
        Apply the Macklin complementarity preconditioner.
    """

    def __init__(
        self,
        maxiter: int = 20,
        tol: float = 1e-10,
        pcr_maxiter: int = 100,
        pcr_tol: float = 1e-10,
        damped_step_fraction: float = 0.75,
        diagonal_regularization: float = 0.0,
        use_preconditioner: bool = True,
    ):
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.pcr_maxiter = int(pcr_maxiter)
        self.pcr_tol = float(pcr_tol)
        self.damped_step_fraction = float(damped_step_fraction)
        self.diagonal_regularization = float(diagonal_regularization)
        self.use_preconditioner = bool(use_preconditioner)

    def _to_dense(self, M):
        if sp.issparse(M):
            return M.toarray()
        return np.asarray(M, dtype=float)

    def _compute_H_inv_action(self, H):
        """Return a callable v -> H^{-1} v.

        Dense LU for small systems; sparse SPLU for large.
        """
        n = H.shape[0]
        if sp.issparse(H) and n > 200:
            H_lu = spla.splu(H.tocsc())
            return lambda v: H_lu.solve(v)
        else:
            H_dense = self._to_dense(H)
            H_lu_piv = la.lu_factor(H_dense)
            return lambda v, _lu=H_lu_piv: la.lu_solve(_lu, v)

    def _compute_preconditioner(self, J, H_inv_action):
        """Diagonal preconditioner r_i = [J H^{-1} J^T]_{ii}."""
        J_dense = self._to_dense(J)
        m = J_dense.shape[0]
        diag = np.empty(m, dtype=float)
        for i in range(m):
            row = J_dense[i, :]
            diag[i] = float(np.dot(row, H_inv_action(row)))
        diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
        inv_diag = 1.0 / diag
        return lambda v: inv_diag * v

    def solve_linear(
        self,
        H: np.ndarray,
        J: np.ndarray,
        C: np.ndarray,
        g: np.ndarray,
        h_c: np.ndarray,
        precond_diag: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Solve one linearized saddle-point system via Schur complement.

        Parameters
        ----------
        H : (n_phys, n_phys)
        J : (n_react, n_phys)
        C : (n_react, n_react)
        g : (n_phys,)  momentum residual
        h_c : (n_react,)  contact residual
        precond_diag : (n_react,) or None

        Returns
        -------
        delta_u : (n_phys,)
        delta_lam : (n_react,)
        info : dict with 'converged', 'iterations', 'residual_norm'
        """
        g = np.asarray(g, dtype=float).ravel()
        h_c = np.asarray(h_c, dtype=float).ravel()
        J_arr = self._to_dense(J)
        C_arr = self._to_dense(C)

        H_inv = self._compute_H_inv_action(H)

        H_inv_g = H_inv(g)
        schur_rhs = h_c - J_arr @ H_inv_g

        def schur_matvec(v):
            Jt_v = J_arr.T @ v
            H_inv_Jt_v = H_inv(Jt_v)
            return J_arr @ H_inv_Jt_v + C_arr @ v

        precond = None
        if self.use_preconditioner:
            if precond_diag is not None:
                safe = np.where(np.abs(precond_diag) > 1e-30, precond_diag, 1.0)
                precond = lambda v: v / safe
            else:
                precond = self._compute_preconditioner(J_arr, H_inv)

        eps = self.diagonal_regularization
        if eps > 0.0:
            _base_matvec = schur_matvec

            def schur_matvec(v, _m=_base_matvec, _e=eps):
                return _m(v) + _e * v

        delta_lam, pcr_info = pcr_solve(
            schur_matvec,
            schur_rhs,
            maxiter=self.pcr_maxiter,
            tol=self.pcr_tol,
            preconditioner=precond,
        )

        delta_u = H_inv(g + J_arr.T @ delta_lam)

        return delta_u, delta_lam, pcr_info

    def solve(
        self,
        block_system: BlockStructuredSystem,
        y0: np.ndarray,
        t: float,
        h: float,
        y_prev: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, int]:
        """Run the full Newton loop using block-structured assembly.

        Parameters
        ----------
        block_system : BlockStructuredSystem
        y0 : ndarray
            Initial guess for augmented state [u; lambda].
        t : float
            Time at the end of the step.
        h : float
            Step size.
        y_prev : ndarray
            State at the beginning of the step.

        Returns
        -------
        y_new : ndarray
        err : float
        converged : bool
        iterations : int
        """
        n_p = block_system.n_phys
        n_r = block_system.n_react
        y = np.asarray(y0, dtype=float).ravel().copy()

        for iteration in range(1, self.maxiter + 1):
            blocks = block_system.assemble_blocks(y, t, h, y_prev)
            H = blocks["H"]
            J = blocks["J"]
            C = blocks["C"]
            g = blocks["g"]
            h_c = blocks["h_c"]
            precond_diag = blocks.get("precond_diag", None)

            err_g = float(np.linalg.norm(g))
            err_h = float(np.linalg.norm(h_c))
            err = max(err_g, err_h)

            if err < self.tol:
                return y.copy(), err, True, iteration

            delta_u, delta_lam, pcr_info = self.solve_linear(
                H, J, C, g, h_c, precond_diag,
            )

            alpha = self.damped_step_fraction
            y[:n_p] += alpha * delta_u
            y[n_p : n_p + n_r] += alpha * delta_lam

        blocks = block_system.assemble_blocks(y, t, h, y_prev)
        err = max(
            float(np.linalg.norm(blocks["g"])),
            float(np.linalg.norm(blocks["h_c"])),
        )
        return y.copy(), err, False, self.maxiter
