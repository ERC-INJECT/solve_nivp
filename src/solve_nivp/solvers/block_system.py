"""Block-structured Newton solver using Schur-complement reduction.

Generalises the compliance-formulation Newton method from Macklin et al.
(2019) §6 to non-symmetric off-diagonal blocks.  The 2×2 block system

    [H,      B_top] [Δu]   [g]
    [B_bot,    C  ] [Δλ] = [h_c]

is reduced via Schur complement

    S = C - B_bot H⁻¹ B_top
    S Δλ = h_c - B_bot H⁻¹ g
    Δu   = H⁻¹ (g - B_top Δλ)

When B_top = -J^T and B_bot = J (Macklin velocity-level constraints)
this recovers S = J H⁻¹ J^T + C.  For position-level NCP constraints
the off-diagonal blocks differ and must be kept separate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .pcr import pcr_solve


@runtime_checkable
class BlockStructuredSystem(Protocol):
    """Protocol for systems that expose 4-block Newton Jacobian."""

    n_phys: int
    n_react: int

    def assemble_blocks(
        self,
        y: np.ndarray,
        t: float,
        h: float,
        y_prev: np.ndarray,
    ) -> Dict[str, Any]:
        """Return the block components for the Newton system.

        Returns
        -------
        dict with keys:
            H : ndarray or sparse, (n_phys, n_phys)
            B_top : ndarray or sparse, (n_phys, n_react)
            B_bot : ndarray or sparse, (n_react, n_phys)
            C : ndarray or sparse, (n_react, n_react)
            g : ndarray, (n_phys,)
            h_c : ndarray, (n_react,)
            precond_diag : ndarray, (n_react,) or None
        """
        ...


class SchurComplementSolver:
    """Newton solver using Schur-complement reduction.

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
        Macklin §8.1 damped Newton step alpha in (0, 1].
    diagonal_regularization : float
        Macklin §8.4 epsilon added to Schur diagonal.
    use_preconditioner : bool
        Apply the Macklin complementarity preconditioner.
    linear_solver : str
        ``"direct"`` assembles the full saddle-point matrix and uses
        dense LU / sparse SPLU.  ``"pcr"`` uses the Schur complement
        with PCR.
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
        linear_solver: str = "direct",
    ):
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.pcr_maxiter = int(pcr_maxiter)
        self.pcr_tol = float(pcr_tol)
        self.damped_step_fraction = float(damped_step_fraction)
        self.diagonal_regularization = float(diagonal_regularization)
        self.use_preconditioner = bool(use_preconditioner)
        self.linear_solver = str(linear_solver).strip().lower()
        if self.linear_solver not in ("direct", "pcr"):
            raise ValueError(
                f"linear_solver must be 'direct' or 'pcr', got {linear_solver!r}"
            )

    def _to_dense(self, M):
        if sp.issparse(M):
            return M.toarray()
        return np.asarray(M, dtype=float)

    def _compute_H_inv_action(self, H):
        """Return a callable v -> H^{-1} v."""
        n = H.shape[0]
        if sp.issparse(H) and n > 200:
            H_lu = spla.splu(H.tocsc())
            return lambda v: H_lu.solve(v)
        else:
            H_dense = self._to_dense(H)
            H_lu_piv = la.lu_factor(H_dense)
            return lambda v, _lu=H_lu_piv: la.lu_solve(_lu, v)

    def _solve_linear_direct(self, H, B_top, B_bot, C, g, h_c):
        """Solve the full saddle-point system with a direct factorisation."""
        n_p = g.shape[0]
        rhs_full = np.concatenate([g, h_c])

        if any(sp.issparse(M) for M in (H, B_top, B_bot, C)):
            A_full = sp.bmat([[H, B_top],
                              [B_bot, C]], format="csr")
            x = spla.spsolve(A_full, rhs_full)
            residual = float(np.linalg.norm(A_full @ x - rhs_full))
        else:
            A_full = np.block([[self._to_dense(H), self._to_dense(B_top)],
                               [self._to_dense(B_bot), self._to_dense(C)]])
            x = la.solve(A_full, rhs_full)
            residual = float(np.linalg.norm(A_full @ x - rhs_full))
        return x[:n_p], x[n_p:], {
            "converged": True, "iterations": 1,
            "residual_norm": residual, "residual_history": [residual],
        }

    def _solve_linear_pcr(self, H, B_top, B_bot, C, g, h_c, precond_diag):
        """Solve via Schur complement S = C - B_bot H^{-1} B_top with PCR."""
        Bt = self._to_dense(B_top)
        Bb = self._to_dense(B_bot)
        C_arr = self._to_dense(C)

        H_inv = self._compute_H_inv_action(H)
        schur_rhs = h_c - Bb @ H_inv(g)

        def schur_matvec(v):
            return C_arr @ v - Bb @ H_inv(Bt @ v)

        precond = None
        if self.use_preconditioner and precond_diag is not None:
            safe = np.where(np.abs(precond_diag) > 1e-30, precond_diag, 1.0)
            precond = lambda v: v / safe

        eps = self.diagonal_regularization
        if eps > 0.0:
            _base = schur_matvec
            def schur_matvec(v, _m=_base, _e=eps):
                return _m(v) + _e * v

        delta_lam, pcr_info = pcr_solve(
            schur_matvec, schur_rhs,
            maxiter=self.pcr_maxiter, tol=self.pcr_tol,
            preconditioner=precond,
        )
        delta_u = H_inv(g - Bt @ delta_lam)
        return delta_u, delta_lam, pcr_info

    def solve_linear(
        self,
        H: np.ndarray,
        B_top: np.ndarray,
        B_bot: np.ndarray,
        C: np.ndarray,
        g: np.ndarray,
        h_c: np.ndarray,
        precond_diag: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Solve one linearized saddle-point system.

        Parameters
        ----------
        H : (n_phys, n_phys)
        B_top : (n_phys, n_react)
        B_bot : (n_react, n_phys)
        C : (n_react, n_react)
        g : (n_phys,)
        h_c : (n_react,)
        precond_diag : (n_react,) or None

        Returns
        -------
        delta_u, delta_lam, info
        """
        g = np.asarray(g, dtype=float).ravel()
        h_c = np.asarray(h_c, dtype=float).ravel()

        if self.linear_solver == "direct":
            return self._solve_linear_direct(H, B_top, B_bot, C, g, h_c)
        else:
            return self._solve_linear_pcr(
                H, B_top, B_bot, C, g, h_c, precond_diag,
            )

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
            Initial guess [u; lambda].
        t, h, y_prev : float, float, ndarray
        """
        n_p = block_system.n_phys
        n_r = block_system.n_react
        y = np.asarray(y0, dtype=float).ravel().copy()

        for iteration in range(1, self.maxiter + 1):
            blocks = block_system.assemble_blocks(y, t, h, y_prev)
            H = blocks["H"]
            B_top = blocks["B_top"]
            B_bot = blocks["B_bot"]
            C = blocks["C"]
            g = blocks["g"]
            h_c = blocks["h_c"]
            precond_diag = blocks.get("precond_diag", None)

            err = max(
                float(np.linalg.norm(g)),
                float(np.linalg.norm(h_c)),
            )
            if err < self.tol:
                return y.copy(), err, True, iteration

            delta_u, delta_lam, info = self.solve_linear(
                H, B_top, B_bot, C, g, h_c, precond_diag,
            )

            if not info.get("converged", True):
                return y.copy(), err, False, iteration

            alpha = self.damped_step_fraction
            y[:n_p] += alpha * delta_u
            y[n_p:n_p + n_r] += alpha * delta_lam

        blocks = block_system.assemble_blocks(y, t, h, y_prev)
        err = max(
            float(np.linalg.norm(blocks["g"])),
            float(np.linalg.norm(blocks["h_c"])),
        )
        return y.copy(), err, False, self.maxiter

    # ------------------------------------------------------------------
    # Multi-stage coupled Newton (RadauIIA)
    # ------------------------------------------------------------------

    def solve_coupled(
        self,
        block_system: BlockStructuredSystem,
        y0: np.ndarray,
        t: float,
        h: float,
        y_prev: np.ndarray,
        rk_A: np.ndarray,
        rk_c: np.ndarray,
        A_phys: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, int]:
        """Coupled Newton for multi-stage Radau IIA with Schur reduction.

        Stacks all *s* stages into a single Newton system.  The velocity
        block ``H_full`` carries Butcher cross-stage coupling; ``B_bot``
        and ``C`` are block-diagonal (each stage's NCP depends only on
        its own state).  ``B_top`` has off-diagonal blocks because the
        physical RHS includes the contact-force coupling ``B·λ``.

        Parameters
        ----------
        block_system : BlockStructuredSystem
        y0 : (n_aug,)
        t : float
            Time at beginning of step.
        h : float
            Full step size.
        y_prev : (n_aug,)
        rk_A : (s, s)
            Butcher A matrix.
        rk_c : (s,)
        A_phys : (n_phys, n_phys)
            Physical mass matrix.

        Returns
        -------
        Y_s, err, converged, iterations
        """
        s = rk_A.shape[0]
        n_p = block_system.n_phys
        n_r = block_system.n_react
        n_aug = n_p + n_r
        sn_p = s * n_p
        sn_r = s * n_r

        A_arr = self._to_dense(A_phys)
        Z = np.tile(y0, s)

        for iteration in range(1, self.maxiter + 1):
            # -- Per-stage block assembly --
            per_stage = []
            f_phys = []
            for i in range(s):
                Y_i = Z[i * n_aug:(i + 1) * n_aug]
                aii = rk_A[i, i]
                stage_h = aii * h
                t_i = t + rk_c[i] * h

                bi = block_system.assemble_blocks(Y_i, t_i, stage_h, y_prev)
                per_stage.append(bi)

                A_over_aih = A_arr / stage_h
                f_i = bi["g"] + A_over_aih @ (Y_i[:n_p] - y_prev[:n_p])
                f_phys.append(f_i)

            # -- Stacked residual --
            g_full = np.empty(sn_p)
            hc_full = np.empty(sn_r)
            for i in range(s):
                aii = rk_A[i, i]
                g_i = per_stage[i]["g"].copy()
                for j in range(s):
                    if j != i:
                        g_i += (rk_A[i, j] / aii) * f_phys[j]
                g_full[i * n_p:(i + 1) * n_p] = g_i
                hc_full[i * n_r:(i + 1) * n_r] = per_stage[i]["h_c"]

            err = max(float(np.linalg.norm(g_full)),
                      float(np.linalg.norm(hc_full)))
            if err < self.tol:
                Y_s = Z[(s - 1) * n_aug:s * n_aug]
                return Y_s.copy(), err, True, iteration

            # -- H_full: (s·n_p × s·n_p) with Butcher cross-stage coupling --
            H_full = np.zeros((sn_p, sn_p))
            for i in range(s):
                aii = rk_A[i, i]
                r0, r1 = i * n_p, (i + 1) * n_p
                for j in range(s):
                    c0, c1 = j * n_p, (j + 1) * n_p
                    if i == j:
                        H_full[r0:r1, c0:c1] = self._to_dense(per_stage[i]["H"])
                    else:
                        H_j = self._to_dense(per_stage[j]["H"])
                        dfdu_j = A_arr / (rk_A[j, j] * h) - H_j
                        H_full[r0:r1, c0:c1] = -(rk_A[i, j] / aii) * dfdu_j

            # -- B_top_full: (s·n_p × s·n_r) — NOT block-diagonal --
            Bt_full = np.zeros((sn_p, sn_r))
            for i in range(s):
                aii = rk_A[i, i]
                r0, r1 = i * n_p, (i + 1) * n_p
                for j in range(s):
                    c0, c1 = j * n_r, (j + 1) * n_r
                    if i == j:
                        Bt_full[r0:r1, c0:c1] = self._to_dense(
                            per_stage[i]["B_top"],
                        )
                    else:
                        Bt_full[r0:r1, c0:c1] = (
                            (rk_A[i, j] / aii)
                            * self._to_dense(per_stage[j]["B_top"])
                        )

            # -- B_bot_diag, C_diag: block-diagonal --
            Bb_diag = np.zeros((sn_r, sn_p))
            C_diag = np.zeros((sn_r, sn_r))
            for i in range(s):
                Bb_diag[i * n_r:(i + 1) * n_r,
                        i * n_p:(i + 1) * n_p] = self._to_dense(
                    per_stage[i]["B_bot"],
                )
                C_diag[i * n_r:(i + 1) * n_r,
                       i * n_r:(i + 1) * n_r] = self._to_dense(
                    per_stage[i]["C"],
                )

            # -- Solve the stacked saddle-point system --
            delta_u, delta_lam, info = self.solve_linear(
                H_full, Bt_full, Bb_diag, C_diag, g_full, hc_full,
            )

            if not info.get("converged", True):
                Y_s = Z[(s - 1) * n_aug:s * n_aug]
                return Y_s.copy(), err, False, iteration

            alpha = self.damped_step_fraction
            for i in range(s):
                Z[i * n_aug:i * n_aug + n_p] += (
                    alpha * delta_u[i * n_p:(i + 1) * n_p]
                )
                Z[i * n_aug + n_p:(i + 1) * n_aug] += (
                    alpha * delta_lam[i * n_r:(i + 1) * n_r]
                )

        Y_s = Z[(s - 1) * n_aug:s * n_aug]
        return Y_s.copy(), err, False, self.maxiter
