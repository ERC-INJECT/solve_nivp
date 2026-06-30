"""Macklin et al. (2019) velocity-level contact for Schur-complement Newton.

Reformulates the normal Signorini condition from position-level gap(q) >= 0
to velocity-level  J*v + g_prev/h >= 0,  giving symmetric off-diagonal
blocks in the Newton saddle-point system.  Friction uses the Macklin
compliance law (same as ncp_contact.py).

The ``MacklinBlockSystem`` satisfies the ``BlockStructuredSystem`` protocol
and plugs directly into ``BackwardEulerSchur`` / ``SchurComplementSolver``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

from .ncp_contact import (
    _contact_block_residual_and_jac,
    _normalize_ncp_name,
)


class MacklinBlockSystem:
    """Velocity-level block-structured contact system (Macklin 2019 S4-6).

    Parameters
    ----------
    A_phys : ndarray, (n_phys, n_phys)
    rhs_smooth : callable
        ``f(t, y) -> (n_phys,)`` smooth-physics RHS.
    n_phys, n_react : int
    contacts : list of dict
        Normalised contacts with keys ``vN``, ``vT``, ``mu``.
    gap_func : callable
        ``gap(y_phys, t) -> (n_contacts,)``.
    B_mat : ndarray, (n_phys, n_react)
    vel_indices : ndarray of int
    rhs_jac : callable or None
    ncp_type, friction_law : str
    normal_r, friction_r : float
    """

    def __init__(
        self,
        A_phys,
        rhs_smooth,
        n_phys: int,
        n_react: int,
        contacts: list,
        gap_func,
        B_mat,
        vel_indices,
        rhs_jac=None,
        ncp_type: str = "fischer_burmeister",
        friction_law: str = "compliance",
        normal_r: float = 1.0,
        friction_r: float = 1.0,
    ):
        self.n_phys = n_phys
        self.n_react = n_react
        self._A = np.asarray(A_phys, dtype=float)
        self._rhs_smooth = rhs_smooth
        self._rhs_jac = rhs_jac
        self._contacts = contacts
        self._gap_func = gap_func
        self._B_mat = np.asarray(B_mat, dtype=float)
        self._vel_indices = np.asarray(vel_indices, dtype=int)
        self._ncp_type = ncp_type
        self._friction_law = friction_law
        self._normal_r = float(normal_r)
        self._friction_r = float(friction_r)

    def assemble_blocks(
        self,
        y: np.ndarray,
        t: float,
        h: float,
        y_prev: np.ndarray,
    ) -> Dict[str, Any]:
        """Assemble the 2x2 block Newton system at velocity level.

        The normal constraint uses ``c_N = v_N + gap_prev / h``
        instead of ``gap(q)``, giving ``B_top = -B_bot^T``.
        """
        n_p = self.n_phys
        n_r = self.n_react
        z = y[:n_p]
        r = y[n_p:]
        z_prev = y_prev[:n_p]

        A_over_h = self._A / h

        if self._rhs_jac is not None:
            J_smooth = np.asarray(self._rhs_jac(t, z), dtype=float)
        else:
            J_smooth = self._fd_smooth_jac(t, z)
        H = A_over_h - J_smooth

        B_top = -(1.0 / h) * self._B_mat

        f_smooth = np.asarray(self._rhs_smooth(t, z), dtype=float).ravel()
        phys_residual = (
            A_over_h @ (z - z_prev) - f_smooth - (1.0 / h) * self._B_mat @ r
        )
        g = -phys_residual

        gap_prev = np.atleast_1d(self._gap_func(z_prev, t - h))

        B_bot = np.zeros((n_r, n_p), dtype=float)
        C = np.zeros((n_r, n_r), dtype=float)
        h_c = np.zeros(n_r, dtype=float)

        col = 0
        for k, ci in enumerate(self._contacts):
            vN = ci["vN"]
            vT = ci["vT"]
            d = 1 + len(vT)
            sl = slice(col, col + d)

            r_blk = r[sl]

            c_N = float(z[vN]) + float(gap_prev[k]) / h

            u_blk = np.zeros(d, dtype=float)
            u_blk[0] = c_N
            for j, vt in enumerate(vT):
                u_blk[1 + j] = z[vt]

            mu_k = ci["mu"]
            f_blk, df_dgap, df_du, df_dr = _contact_block_residual_and_jac(
                c_N,
                u_blk,
                r_blk,
                mu_k,
                self._ncp_type,
                self._ncp_type,
                self._normal_r,
                self._friction_r,
                self._friction_law,
            )

            h_c[sl] = f_blk

            # B_bot: derivative of NCP residual w.r.t. physical DOFs.
            # Normal row: dc_N/dz[vN] = 1  (velocity-level, not position-level)
            # So ∂f_blk[0]/∂z[vN] = df_dgap[0] * 1 + df_du[0, 0] * 1
            B_bot[col, vN] = -(df_dgap[0] + df_du[0, 0])
            for j, vt in enumerate(vT):
                B_bot[col, vt] = -df_du[0, 1 + j]
                B_bot[col + 1 + j, vN] = -df_du[1 + j, 0]
                for jj, vt2 in enumerate(vT):
                    B_bot[col + 1 + j, vt2] = -df_du[1 + j, 1 + jj]

            C[sl, sl] = -df_dr

            col += d

        H_diag = np.diag(H)
        safe_diag = np.where(np.abs(H_diag) > 1e-30, H_diag, 1.0)
        precond_diag = np.array([
            float(np.sum(B_bot[i, :] ** 2 / safe_diag))
            for i in range(n_r)
        ])
        precond_diag = np.where(precond_diag > 1e-30, precond_diag, 1.0)

        return {
            "H": H,
            "B_top": B_top,
            "B_bot": B_bot,
            "C": C,
            "g": g,
            "h_c": h_c,
            "precond_diag": precond_diag,
        }

    def _fd_smooth_jac(self, t, z):
        f0 = np.asarray(self._rhs_smooth(t, z), dtype=float).ravel()
        n = z.size
        eps = 1e-7
        h_vec = eps * np.maximum(np.abs(z), 1.0)
        J = np.empty((n, n), dtype=float)
        for j in range(n):
            z_p = z.copy()
            z_p[j] += h_vec[j]
            fp = np.asarray(self._rhs_smooth(t, z_p), dtype=float).ravel()
            J[:, j] = (fp - f0) / h_vec[j]
        return J


def build_macklin_contact_blocked(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func,
    B=None,
    rhs_jac=None,
    ncp_type="fischer_burmeister",
    friction_law="compliance",
    normal_r=1.0,
    friction_r=1.0,
):
    """Build a Macklin (2019) velocity-level block system for Schur solve.

    Parameters
    ----------
    A : ndarray or sparse, (n_phys, n_phys)
        Physical mass / descriptor matrix.
    rhs_smooth : callable
        ``f(t, y) -> (n_phys,)`` smooth forces.
    y0 : ndarray, (n_phys,)
        Physical initial condition.
    contacts : list of dict
        Each with ``vel_normal_idx``, ``vel_tangential_idx``, ``mu``.
    gap_func : callable
        ``gap(y_phys, t) -> (n_contacts,)``.
    B : ndarray or sparse or None
        Reaction coupling; built from contacts if None.
    rhs_jac : callable or None
    ncp_type, friction_law : str
    normal_r, friction_r : float

    Returns
    -------
    MacklinBlockSystem
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size
    ncp_type = _normalize_ncp_name(ncp_type, label="ncp_type")

    norm_contacts = []
    vel_indices_list = []
    n_react = 0
    for c in contacts:
        vN = int(c["vel_normal_idx"])
        vT = list(np.atleast_1d(c.get("vel_tangential_idx", [])).astype(int))
        mu_val = c.get("mu", 0.0)
        if callable(mu_val):
            raise TypeError(
                "build_macklin_contact_blocked does not support callable "
                "(state-dependent) mu; pass a scalar friction coefficient")
        mu = float(mu_val)
        norm_contacts.append({"vN": vN, "vT": vT, "mu": mu})
        vel_indices_list.append(vN)
        vel_indices_list.extend(vT)
        n_react += 1 + len(vT)

    vel_indices = np.asarray(vel_indices_list, dtype=int)

    if B is not None:
        B_mat = B.toarray() if sp.issparse(B) else np.asarray(B, dtype=float)
    else:
        B_mat = np.zeros((n_phys, n_react), dtype=float)
        col = 0
        for ci in norm_contacts:
            B_mat[ci["vN"], col] = 1.0
            col += 1
            for vt in ci["vT"]:
                B_mat[vt, col] = 1.0
                col += 1

    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)

    return MacklinBlockSystem(
        A_phys=A_dense,
        rhs_smooth=rhs_smooth,
        n_phys=n_phys,
        n_react=n_react,
        contacts=norm_contacts,
        gap_func=gap_func,
        B_mat=B_mat,
        vel_indices=vel_indices,
        rhs_jac=rhs_jac,
        ncp_type=ncp_type,
        friction_law=friction_law,
        normal_r=normal_r,
        friction_r=friction_r,
    )
