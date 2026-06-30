"""Dynamic De Saxce cone contact backend.

This module implements a true zero-restitution De Saxce / Moreau contact
solve in contact space rather than splitting the cone law across explicit
reaction residual rows and a separate SOC projection.

The builder returns an augmented system ``[y_phys, r]`` where the reaction
block ``r`` is stored for diagnostics only.  The physical RHS remains the
smooth dynamic system; the projection solves the active contact problem

    R = Proj_{K_mu}(R - rho * u_hat(R))

on the active contact set and injects the resulting reaction back into the
physical state through the linearized implicit step response.

Only the zero-restitution case is supported here because that is the only
mode used by the benchmark suite.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .alart_curnier_contact import (
    _call_state_block_time_fk,
    _call_state_time_fk,
    _call_with_time_state_fk,
    _count_required_args,
    _dense_or_sparse,
    _eval_s0,
    _eval_w0,
    _parse_prev_and_h,
)
from .contact import ContactSystem
from .projections import (
    AlgebraicConstraintProjection,
    IdentityProjection,
    MuScaledSOCProjection,
    Projection,
)


class _DeSaxceConeProjection(Projection):
    """Contact-space De Saxce cone solve."""

    def __init__(
        self,
        *,
        n_phys,
        n_react,
        block_slices,
        gap,
        local_model_builder,
        reaction_offset=None,
        gap_tol=0.0,
        inactive_handling="gap",
        prox_rho="auto",
        prox_rho_scale=1.0,
        prox_rho_min=1.0e-8,
        prox_rho_max=1.0e8,
        vi_maxit=30,
        vi_tol=1.0e-10,
        tc_tol=1.0e-10,
        component_slices=None,
    ):
        super().__init__(component_slices=component_slices)
        self.n_phys = int(n_phys)
        self.n_react = int(n_react)
        self.reaction_offset = (
            None if reaction_offset is None else int(reaction_offset)
        )
        self.block_slices = list(block_slices)
        self.n_blocks = len(self.block_slices)
        self.gap = gap
        self.local_model_builder = local_model_builder
        self.gap_tol = float(gap_tol)
        _ih = str(inactive_handling).strip().lower().replace("-", "_")
        if _ih not in {"gap", "cone"}:
            raise ValueError(
                "inactive_handling must be 'gap' (geometric gate) or 'cone' "
                f"(let the cone projection decide); got {inactive_handling!r}"
            )
        self.inactive_handling = _ih
        self.prox_rho = prox_rho
        self.prox_rho_scale = float(prox_rho_scale)
        self.prox_rho_min = float(prox_rho_min)
        self.prox_rho_max = float(prox_rho_max)
        self.vi_maxit = int(vi_maxit)
        self.vi_tol = float(vi_tol)
        self.tc_tol = float(tc_tol)
        self._gap_callback_nargs = _count_required_args(gap)
        self._gap_nargs = (
            1
            if self._gap_callback_nargs is not None and self._gap_callback_nargs <= 1
            else 2
        )
        self.gap_func = lambda y, t=None: self._gap_values(t, y)
        self._locked_active = None
        self.reconcile_locked_active = False
        self._last_inner_info = None
        self._last_reaction_full = np.zeros(self.n_react, dtype=float)
        self._last_solve_data = None

    def _gap_values(self, t, y):
        if self._gap_callback_nargs is not None and self._gap_callback_nargs <= 1:
            return np.asarray(self.gap(y), dtype=float)
        return np.asarray(self.gap(y, t), dtype=float)

    def _active_blocks(self, t, y_bar, y_it):
        if self._locked_active is not None:
            return np.flatnonzero(self._locked_active)
        if self.inactive_handling == "cone":
            # All blocks always active; the cone projection P_{K_mu} naturally
            # returns r=0 for open contacts (u_hat_N > 0 maps outside the cone).
            return np.arange(self.n_blocks, dtype=int)
        g_bar = self._gap_values(t, y_bar)
        g_it = self._gap_values(t, y_it)
        return np.where(np.minimum(g_bar, g_it) <= self.gap_tol)[0]

    def reset_branch_cache(self):
        return None

    def lock_active_set(
        self, y, t=None, candidate=None, step_size=None, reset_branch=True
    ):
        if self.inactive_handling == "cone":
            # All blocks active — cone decides during the inner solve.
            self._locked_active = np.ones(self.n_blocks, dtype=bool)
            return
        gaps = self._gap_values(t, np.asarray(y, dtype=float))
        if candidate is not None:
            gaps = np.minimum(gaps, self._gap_values(t, np.asarray(candidate, dtype=float)))
        self._locked_active = gaps <= self.gap_tol

    def unlock_active_set(self):
        self._locked_active = None

    @staticmethod
    def _project_blocks(z, mu_blocks, block_slices, return_jacobian=False):
        z = np.asarray(z, dtype=float)
        if not return_jacobian:
            out = np.zeros_like(z)
            for k, sl in enumerate(block_slices):
                out[sl] = MuScaledSOCProjection._proj_mu_scaled_soc(
                    z[sl], float(mu_blocks[k]), return_jacobian=False
                )
            return out

        out = np.zeros_like(z)
        J = np.zeros((z.size, z.size), dtype=float)
        for k, sl in enumerate(block_slices):
            p_blk, J_blk = MuScaledSOCProjection._proj_mu_scaled_soc(
                z[sl], float(mu_blocks[k]), return_jacobian=True
            )
            out[sl] = p_blk
            J[sl, sl] = J_blk
        return out, J

    @staticmethod
    def _uhat_and_jac(u, alpha_blocks, block_slices, *, tie_tol=1.0e-14):
        u = np.asarray(u, dtype=float)
        u_hat = u.copy()
        D = np.eye(u.size, dtype=float)

        for k, sl in enumerate(block_slices):
            alpha_k = float(alpha_blocks[k])
            if sl.stop - sl.start <= 1 or abs(alpha_k) <= tie_tol:
                continue
            u_t = u[sl.start + 1 : sl.stop]
            norm_ut = float(np.linalg.norm(u_t))
            u_hat[sl.start] += alpha_k * norm_ut
            if norm_ut > tie_tol:
                D[sl.start, sl.start + 1 : sl.stop] = alpha_k * (u_t / norm_ut)
            else:
                D[sl.start, sl.start + 1 : sl.stop] = 0.0
        return u_hat, D

    def _expand_block_values(self, block_vals, block_slices):
        out = np.zeros(sum(sl.stop - sl.start for sl in block_slices), dtype=float)
        for k, sl in enumerate(block_slices):
            out[sl] = float(block_vals[k])
        return out

    def _block_rho(self, W, block_slices):
        n_blocks = len(block_slices)
        if isinstance(self.prox_rho, str):
            if self.prox_rho.strip().lower() != "auto":
                raise ValueError(
                    "prox_rho must be 'auto', a scalar, or an array of block values"
                )
            rho = np.empty(n_blocks, dtype=float)
            for k, sl in enumerate(block_slices):
                W_blk = np.asarray(W[sl, sl], dtype=float)
                scale = float(np.linalg.norm(W_blk, ord=2))
                scale = max(scale, self.prox_rho_min)
                rho[k] = np.clip(
                    self.prox_rho_scale / scale,
                    self.prox_rho_min,
                    self.prox_rho_max,
                )
            return rho

        rho_arr = np.atleast_1d(np.asarray(self.prox_rho, dtype=float))
        if rho_arr.size == 1:
            return np.full(n_blocks, float(rho_arr[0]), dtype=float)
        if rho_arr.size != n_blocks:
            raise ValueError(
                f"prox_rho must have one value or {n_blocks} values (got {rho_arr.size})"
            )
        return rho_arr.ravel()

    def _natural_map_state(self, R, *, u_free, W, mu, alpha, block_slices, rho_full,
                           offset=None):
        u = u_free + W @ R
        u_hat, D_uhat = self._uhat_and_jac(u, alpha, block_slices)
        R_eff = R if offset is None else R + offset
        z = R_eff - rho_full * u_hat
        proj_z, J_proj = self._project_blocks(
            z, mu, block_slices, return_jacobian=True
        )
        F = R_eff - proj_z
        return F, u, u_hat, D_uhat, J_proj

    @staticmethod
    def _dproj_dmu_block(z_blk, mu):
        r"""Analytical :math:`\partial \Pi_{K_\mu}(z)/\partial\mu` for one block.

        Returns
        -------
        dp : ndarray, shape ``(d,)``
        """
        z_blk = np.asarray(z_blk, dtype=float)
        d = z_blk.size
        if mu <= 0.0:
            return np.zeros(d, dtype=float)
        s = float(z_blk[0])
        w = z_blk[1:]
        r = float(np.linalg.norm(w))
        lam_plus = s + mu * r
        lam_minus = s - r / mu
        if (lam_minus >= 0.0 and s >= 0.0) or lam_plus <= 0.0:
            return np.zeros(d, dtype=float)
        a = 1.0 / (1.0 + mu * mu)
        a2 = a * a
        dp = np.empty(d, dtype=float)
        dp[0] = a2 * (r * (1.0 - mu * mu) - 2.0 * mu * s)
        if r > 0.0:
            dp[1:] = a2 * (s * (1.0 - mu * mu) + 2.0 * mu * r) * (w / r)
        else:
            dp[1:] = 0.0
        return dp

    def _solve_local_problem(self, local_model):
        u_free = np.asarray(local_model["u_free"], dtype=float)
        U_y = np.asarray(local_model["U_y"], dtype=float)
        G_state = np.asarray(local_model["G_state"], dtype=float)
        mu = np.asarray(local_model["mu"], dtype=float)
        alpha = np.asarray(local_model["alpha"], dtype=float)
        block_slices = list(local_model["block_slices"])
        warm_start = np.asarray(local_model["warm_start"], dtype=float)
        offset = local_model.get("offset")
        if offset is not None:
            offset = np.asarray(offset, dtype=float)

        if u_free.size == 0:
            return {
                "reaction": np.zeros(0, dtype=float),
                "u_free": u_free,
                "U_y": U_y,
                "G_state": G_state,
                "W": np.zeros((0, 0), dtype=float),
                "mu": mu,
                "alpha": alpha,
                "block_slices": block_slices,
                "rho_full": np.zeros(0, dtype=float),
                "info": {"success": True, "iterations": 0, "residual": 0.0},
            }

        W = np.asarray(U_y @ G_state, dtype=float)
        rho_blocks = self._block_rho(W, block_slices)
        rho_full = self._expand_block_values(rho_blocks, block_slices)
        if offset is not None:
            R = self._project_blocks(
                warm_start + offset, mu, block_slices, return_jacobian=False
            ) - offset
        else:
            R = self._project_blocks(warm_start, mu, block_slices, return_jacobian=False)

        _nm_kw = dict(u_free=u_free, W=W, mu=mu, alpha=alpha,
                      block_slices=block_slices, rho_full=rho_full, offset=offset)

        info = {"success": False, "iterations": 0, "residual": np.inf}
        for iteration in range(1, self.vi_maxit + 1):
            F, _, u_hat, D_uhat, J_proj = self._natural_map_state(R, **_nm_kw)
            res = float(np.linalg.norm(F, ord=np.inf))
            info["iterations"] = iteration
            info["residual"] = res
            if res <= self.vi_tol * max(1.0, np.linalg.norm(R, ord=np.inf)):
                info["success"] = True
                break

            rhoD = rho_full[:, None] * D_uhat
            J_nat = np.eye(R.size, dtype=float) - J_proj @ (
                np.eye(R.size, dtype=float) - rhoD @ W
            )

            try:
                dR = np.linalg.solve(J_nat, -F)
            except np.linalg.LinAlgError:
                reg = 1.0e-12 * max(1.0, np.linalg.norm(J_nat, ord=2))
                dR = np.linalg.solve(J_nat + reg * np.eye(J_nat.shape[0]), -F)

            base_merit = 0.5 * float(F @ F)
            accepted = False
            step = 1.0
            for _ in range(10):
                R_trial = R + step * dR
                F_trial, _, _, _, _ = self._natural_map_state(R_trial, **_nm_kw)
                merit_trial = 0.5 * float(F_trial @ F_trial)
                if np.isfinite(merit_trial) and merit_trial < base_merit:
                    R = R_trial
                    accepted = True
                    break
                step *= 0.5

            if accepted:
                continue

            R_eff = R if offset is None else R + offset
            R_next = self._project_blocks(
                R_eff - rho_full * u_hat, mu, block_slices, return_jacobian=False
            )
            if offset is not None:
                R_next = R_next - offset
            if np.linalg.norm(R_next - R, ord=np.inf) <= self.vi_tol * max(
                1.0, np.linalg.norm(R, ord=np.inf)
            ):
                R = R_next
                info["success"] = True
                break
            R = R_next

        F_c, u_c, u_hat_c, D_uhat_c, J_proj_c = self._natural_map_state(
            R, **_nm_kw
        )
        rhoD_c = rho_full[:, None] * D_uhat_c
        I_r = np.eye(R.size, dtype=float)
        J_nat_c = I_r - J_proj_c @ (I_r - rhoD_c @ W)

        return {
            "reaction": R,
            "u_free": u_free,
            "U_y": U_y,
            "G_state": G_state,
            "W": W,
            "mu": mu,
            "alpha": alpha,
            "block_slices": block_slices,
            "rho_full": rho_full,
            "offset": offset,
            "u": u_c,
            "u_hat": u_hat_c,
            "J_proj": J_proj_c,
            "J_nat": J_nat_c,
            "info": info,
        }

    def _embed_full_reaction(self, active_blocks, reaction_active):
        full = np.zeros(self.n_react, dtype=float)
        offset = 0
        for block_idx in np.asarray(active_blocks, dtype=int):
            sl = self.block_slices[block_idx]
            d = sl.stop - sl.start
            full[sl] = np.asarray(reaction_active[offset : offset + d], dtype=float)
            offset += d
        return full

    def _base_state(self, candidate):
        out = np.asarray(candidate, dtype=float).copy()
        if self.reaction_offset is not None:
            out[self.reaction_offset : self.reaction_offset + self.n_react] = 0.0
        return out

    def _solve_full_reaction(
        self, current_state, candidate, *, t=None, Fk_val=None, prev_state=None, step_size=None
    ):
        y_it = np.asarray(current_state, dtype=float)
        y_bar = np.asarray(candidate, dtype=float)
        active_blocks = self._active_blocks(t, y_bar, y_it)
        if active_blocks.size == 0:
            self._last_inner_info = {
                "success": True,
                "iterations": 0,
                "residual": 0.0,
                "active_blocks": 0,
            }
            self._last_reaction_full[:] = 0.0
            self._last_solve_data = None
            return {
                "active_blocks": active_blocks,
                "reaction_active": np.zeros(0, dtype=float),
                "reaction_full": np.zeros(self.n_react, dtype=float),
                "solved": None,
            }

        local_model = self.local_model_builder(
            t=t,
            current_state=y_it,
            candidate=y_bar,
            active_blocks=np.asarray(active_blocks, dtype=int),
            Fk_val=Fk_val,
            prev_state=prev_state,
            step_size=step_size,
            reaction_hint=self._last_reaction_full.copy(),
        )
        solved = self._solve_local_problem(local_model)
        reaction_full = self._embed_full_reaction(active_blocks, solved["reaction"])
        self._last_inner_info = dict(solved["info"])
        self._last_inner_info["active_blocks"] = int(active_blocks.size)
        self._last_reaction_full[:] = reaction_full
        self._last_solve_data = {
            "solved": solved,
            "active_blocks": np.asarray(active_blocks, dtype=int).copy(),
            "y_cur_phys": np.asarray(y_it[:self.n_phys], dtype=float).copy(),
        }
        return {
            "active_blocks": active_blocks,
            "reaction_active": solved["reaction"],
            "reaction_full": reaction_full,
            "solved": solved,
            "y_free": local_model.get("y_free"),
        }

    def reaction_from_state(
        self, state, *, t=None, prev_state=None, step_size=None, Fk_val=None
    ):
        solved = self._solve_full_reaction(
            state,
            state,
            t=t,
            Fk_val=Fk_val,
            prev_state=prev_state,
            step_size=step_size,
        )
        return np.asarray(solved["reaction_full"], dtype=float)

    def project(
        self,
        current_state,
        candidate,
        rhok=None,
        t=None,
        Fk_val=None,
        prev_state=None,
        step_size=None,
        **kw,
    ):
        y_bar = np.asarray(candidate, dtype=float)
        y_base = self._base_state(y_bar)
        solved = self._solve_full_reaction(
            current_state,
            candidate,
            t=t,
            Fk_val=Fk_val,
            prev_state=prev_state,
            step_size=step_size,
        )
        if solved["solved"] is None:
            return y_base
        G_corr = solved["solved"]["G_state"] @ solved["reaction_active"]
        y_free = solved.get("y_free")
        if y_free is not None:
            return np.asarray(y_free, dtype=float) + G_corr
        return y_base + G_corr

    def tangent_cone(
        self,
        candidate,
        current_state,
        rhok=None,
        t=None,
        Fk_val=None,
        prev_state=None,
        step_size=None,
        **kw,
    ):
        n = np.asarray(candidate, dtype=float).size
        return sp.csr_matrix((n, n))

    def tangent_cone_split(
        self,
        candidate,
        current_state,
        rhok=None,
        t=None,
        Fk_val=None,
        prev_state=None,
        step_size=None,
        **kw,
    ):
        r"""Return ``(D_cand, D_state)`` for the semismooth Newton Jacobian.

        ``D_cand = dP/d(candidate) = 0`` when the free trajectory ``y_free``
        is available (the projection output is independent of candidate).

        ``D_state = dP/d(current_state)`` captures the dependence of the
        projection on ``current_state`` through state-dependent friction
        ``mu(y)`` via implicit differentiation of the converged inner solve.
        """
        n = np.asarray(candidate, dtype=float).size
        zero = sp.csr_matrix((n, n))

        data = self._last_solve_data
        if data is None:
            return zero, zero

        solved = data["solved"]
        active_blocks = data["active_blocks"]
        if active_blocks.size == 0 or solved is None:
            return zero, zero

        Dproj = zero

        vcp = getattr(self, "_vectorize_contact_params", None)
        n_phys = getattr(self, "_n_phys", n)
        if vcp is None:
            return Dproj, zero

        y_cur = data["y_cur_phys"]
        mu_0, beta_0 = vcp(y_cur, t=t, Fk_val=Fk_val)
        n_blocks_total = mu_0.size

        eps_fd = 1e-7
        dmu_dy = np.zeros((n_blocks_total, n_phys), dtype=float)
        for j in range(n_phys):
            eps_j = eps_fd * max(1.0, abs(float(y_cur[j])))
            y_pert = y_cur.copy()
            y_pert[j] += eps_j
            mu_pert, _ = vcp(y_pert, t=t, Fk_val=Fk_val)
            col = (mu_pert - mu_0) / eps_j
            if np.any(np.abs(col) > 1e-15):
                dmu_dy[:, j] = col

        if not np.any(np.abs(dmu_dy) > 1e-15):
            return Dproj, zero

        R_0 = solved["reaction"]
        G_state = solved["G_state"]
        mu_active = np.asarray(solved["mu"], dtype=float)
        block_slices = solved["block_slices"]
        rho_full = solved["rho_full"]
        offset = solved.get("offset")
        alpha_active = np.asarray(solved["alpha"], dtype=float)

        block_to_active = {}
        for a_idx, g_idx in enumerate(active_blocks):
            block_to_active[int(g_idx)] = a_idx

        dR_dmu = np.zeros((R_0.size, n_blocks_total), dtype=float)

        for k_global in range(n_blocks_total):
            if k_global not in block_to_active:
                continue
            if np.all(np.abs(dmu_dy[k_global, :]) < 1e-15):
                continue

            k_active = block_to_active[k_global]

            mu_pert = mu_active.copy()
            alpha_pert = alpha_active.copy()
            mu_pert[k_active] -= eps_fd
            alpha_pert[k_active] -= eps_fd

            pert_model = {
                "u_free": solved["u_free"],
                "U_y": solved["U_y"],
                "G_state": G_state,
                "mu": mu_pert,
                "alpha": alpha_pert,
                "block_slices": block_slices,
                "warm_start": R_0.copy(),
                "offset": offset,
            }
            solved_pert = self._solve_local_problem(pert_model)
            dR_dmu[:, k_global] = (R_0 - solved_pert["reaction"]) / eps_fd

        Dstate_dense = G_state @ dR_dmu @ dmu_dy
        return Dproj, sp.csr_matrix(Dstate_dense)


def build_dynamic_desaxce_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    component_slices=None,
    gap_extract=None,
    vel_extract=None,
    constraints=None,
    rhs_jac=None,
    gap_tol=0.0,
    inactive_handling="gap",
    get_s0=None,
    get_w0=None,
    prox_rho="auto",
    prox_rho_scale=1.0,
    prox_rho_min=1.0e-8,
    prox_rho_max=1.0e8,
    vi_maxit=30,
    vi_tol=1.0e-10,
    tc_tol=1.0e-10,
    smooth_rhs_is_affine=False,
):
    r"""Build a dynamic zero-restitution De Saxce cone contact system.

    The returned augmented state is ``[y_phys, r]`` where ``r`` stores the
    force-like contact reaction. The physical RHS remains the smooth dynamic
    system; contact enters through the projection only.

    Parameters
    ----------
    A, rhs_smooth, y0, contacts, gap_func, B, component_slices, constraints,
    rhs_jac
        Same interpretation as in
        :func:`solve_nivp.alart_curnier_contact.build_dynamic_alart_curnier_contact`.
    gap_extract, vel_extract : ndarray or sparse, optional
        Constant extraction operators for the signed gaps and the local
        relative velocities. When ``gap_func`` is omitted, ``gap_extract`` is
        required. ``vel_extract`` should extract the explicit velocity-like
        state components used by the De Saxce law.
    gap_tol : float, default 0.0
        Geometric tolerance for the potentially-active contact set (used only
        when ``inactive_handling="gap"``).
    inactive_handling : {"gap", "cone"}, default "gap"
        Strategy for deciding which contact blocks to project.

        ``"gap"`` (default): a block is active only when its geometric gap
        satisfies ``min(g_bar, g_it) <= gap_tol``.  Open contacts are skipped,
        saving the inner solve.

        ``"cone"``: all blocks are always projected.  The cone projection
        ``P_{K_mu}(R - rho * u_hat)`` returns **zero** for open contacts
        (``u_hat_N > 0`` maps outside the cone) naturally, without any external
        gap gate.  This is the mathematically pure De Saxce bipotential
        formulation.  Costs more per step but avoids geometry-based
        active-set mis-classification near marginal contacts.
    prox_rho : {"auto"} or scalar or array_like, default "auto"
        Proximal parameter used in the natural map

        ``R = Proj_{K_mu}(R - rho * u_hat(R))``.

        ``"auto"`` selects one value per active block from the local block
        response norm.
    prox_rho_scale, prox_rho_min, prox_rho_max
        Tuning knobs for the automatic proximal scaling.
    vi_maxit, vi_tol : int, float
        Inner contact-space natural-map solve controls.
    tc_tol : float
        Tangent-cone tolerance (currently used only for interface symmetry
        with the other projection helpers).

    Notes
    -----
    Only zero restitution is supported here. Any contact dictionary with a
    nonzero ``e`` raises ``ValueError``.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size

    gap_tol = float(gap_tol)

    if gap_extract is not None:
        gap_extract = _dense_or_sparse(gap_extract)
        if gap_extract.shape[1] != n_phys:
            raise ValueError(
                f"gap_extract has {gap_extract.shape[1]} columns but n_phys = {n_phys}"
            )
    if vel_extract is not None:
        vel_extract = _dense_or_sparse(vel_extract)
        if vel_extract.shape[1] != n_phys:
            raise ValueError(
                f"vel_extract has {vel_extract.shape[1]} columns but n_phys = {n_phys}"
            )

    if gap_func is None and gap_extract is None:
        raise ValueError("gap_func must be provided when gap_extract is None")

    norm_contacts = []
    reaction_extract_rows = []
    reaction_idx = 0
    for c in contacts:
        e_val = float(c.get("e", 0.0))
        if abs(e_val) > 1.0e-14:
            raise ValueError(
                "build_dynamic_desaxce_contact only supports zero restitution (e = 0)"
            )

        v_n = int(c["vel_normal_idx"])
        v_t = list(np.atleast_1d(c.get("vel_tangential_idx", [])).astype(int))

        mu_val = c.get("mu", 0.0)
        if callable(mu_val):
            get_mu = mu_val
        else:
            mu_const = float(mu_val)

            def get_mu(y, t=None, Fk_val=None, _m=mu_const):  # noqa: E306
                return _m

        beta_val = c.get("beta", 0.0)
        if callable(beta_val):
            get_beta = beta_val
        else:
            beta_const = float(beta_val)

            def get_beta(y, t=None, Fk_val=None, _b=beta_const):  # noqa: E306
                return _b

        block_slice = slice(reaction_idx, reaction_idx + 1 + len(v_t))
        reaction_extract_rows.extend([v_n] + v_t)
        norm_contacts.append(
            {
                "vN": v_n,
                "vT": v_t,
                "block_slice": block_slice,
                "get_mu": get_mu,
                "mu_nargs": _count_required_args(get_mu),
                "get_beta": get_beta,
                "beta_nargs": _count_required_args(get_beta),
            }
        )
        reaction_idx += 1 + len(v_t)

    n_react = reaction_idx

    if B is None and vel_extract is not None:
        if sp.issparse(vel_extract):
            B_mat = vel_extract[reaction_extract_rows, :].T.tocsr()
        else:
            B_mat = np.asarray(vel_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None and gap_extract is not None:
        if sp.issparse(gap_extract):
            B_mat = gap_extract[reaction_extract_rows, :].T.tocsr()
        else:
            B_mat = np.asarray(gap_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None:
        B_mat = np.zeros((n_phys, n_react), dtype=float)
        col = 0
        for ci in norm_contacts:
            B_mat[ci["vN"], col] = 1.0
            col += 1
            for vt in ci["vT"]:
                B_mat[vt, col] = 1.0
                col += 1
    else:
        B_mat = _dense_or_sparse(B)
        if B_mat.shape != (n_phys, n_react):
            raise ValueError(
                f"B shape {B_mat.shape} doesn't match (n_phys={n_phys}, n_react={n_react})"
            )

    if sp.issparse(A):
        A_phys = A.tocsr()
    else:
        A_phys = np.asarray(A, dtype=float)

    normal_rows = [ci["vN"] for ci in norm_contacts]

    if gap_func is not None:

        def gap_aug(y, t=None):
            return np.atleast_1d(gap_func(np.asarray(y[:n_phys], dtype=float), t))

    else:
        def gap_aug(y, t=None):
            vals = gap_extract @ np.asarray(y[:n_phys], dtype=float)
            vals = np.asarray(vals).ravel()
            return vals[normal_rows]

    if vel_extract is not None:
        U_contact = vel_extract[reaction_extract_rows, :]
        U_contact_csr = (
            U_contact.tocsr() if sp.issparse(U_contact) else sp.csr_matrix(U_contact)
        )
    else:
        vel_indices = np.asarray(reaction_extract_rows, dtype=int)
        U_contact_csr = sp.csr_matrix(
            (
                np.ones(n_react, dtype=float),
                (np.arange(n_react, dtype=int), vel_indices),
            ),
            shape=(n_react, n_phys),
        )

    alg_proj = None
    q_slices = []
    if constraints is not None:
        alg_proj = AlgebraicConstraintProjection(constraints=constraints)
        q_slices = list(alg_proj.constraint_q_slices)

    if component_slices is not None:
        cs_phys = []
        for cs_item in component_slices:
            if isinstance(cs_item, slice):
                cs_phys.append(cs_item)
            else:
                cs_phys.append(np.asarray(cs_item, dtype=int))
    else:
        vel_set = set(reaction_extract_rows)
        vel_idx = np.array(sorted(vel_set), dtype=int)
        other_idx = np.array(sorted(set(range(n_phys)) - vel_set), dtype=int)
        cs_phys = []
        if vel_idx.size > 0:
            cs_phys.append(vel_idx)
        if other_idx.size > 0:
            cs_phys.append(other_idx)

    def _vectorize_contact_params(y_state, *, t=None, Fk_val=None):
        mu = np.empty(len(norm_contacts), dtype=float)
        beta = np.empty(len(norm_contacts), dtype=float)
        for k, ci in enumerate(norm_contacts):
            mu[k] = float(
                _call_state_time_fk(ci["get_mu"], ci["mu_nargs"], y_state, t, Fk_val)
            )
            beta[k] = float(
                _call_state_time_fk(
                    ci["get_beta"], ci["beta_nargs"], y_state, t, Fk_val
                )
            )
        if np.any(beta < -1.0e-14):
            raise ValueError("beta must be nonnegative in build_dynamic_desaxce_contact")
        if np.any(beta > mu + 1.0e-14):
            raise ValueError("beta must satisfy beta <= mu in build_dynamic_desaxce_contact")
        return mu, np.clip(beta, 0.0, mu)

    _has_offset = get_s0 is not None or get_w0 is not None
    _s0_nargs = _count_required_args(get_s0) if callable(get_s0) else None
    _w0_nargs = _count_required_args(get_w0) if callable(get_w0) else None
    n_blocks = len(norm_contacts)

    _B_csr = B_mat.tocsr() if sp.issparse(B_mat) else sp.csr_matrix(B_mat)
    _jac_phys_cache = {}
    _smooth_rhs_affine = [None]
    _smooth_rhs_linear = [None]
    _smooth_rhs_cache_t = [None]
    _response_cache = {}
    _lu_cache = {}

    def _fd_smooth_jac(t, yp):
        f0 = _call_with_time_state_fk(rhs_smooth, t, yp, None)
        eps_base = 1.0e-7
        h_vec = eps_base * np.maximum(np.abs(yp), 1.0)
        J = np.empty((n_phys, n_phys), dtype=float)
        for j in range(n_phys):
            yp_pert = yp.copy()
            yp_pert[j] += h_vec[j]
            fp = _call_with_time_state_fk(rhs_smooth, t, yp_pert, None)
            J[:, j] = (fp - f0) / h_vec[j]
        return J

    def _physical_rhs(t, yp, *, prev_state=None, h_val=None):
        out = np.zeros(n_phys, dtype=float)
        if smooth_rhs_is_affine and rhs_jac is not None:
            t_key = None if t is None else float(t)
            if _smooth_rhs_linear[0] is None or _smooth_rhs_cache_t[0] != t_key:
                J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
                J_s = _dense_or_sparse(J_s)
                _smooth_rhs_linear[0] = (
                    J_s.tocsr() if sp.issparse(J_s) else sp.csr_matrix(J_s)
                )
                _smooth_rhs_affine[0] = np.asarray(
                    _call_with_time_state_fk(rhs_smooth, t, np.zeros(n_phys), None)
                ).ravel()
                _smooth_rhs_cache_t[0] = t_key
            out[:] = np.asarray(_smooth_rhs_linear[0] @ yp).ravel() + _smooth_rhs_affine[0]
        else:
            out[:] = np.asarray(_call_with_time_state_fk(rhs_smooth, t, yp, None)).ravel()

        if alg_proj is not None:
            prev_phys = None if prev_state is None else np.asarray(prev_state[:n_phys], dtype=float)
            c_res = alg_proj.constraint_residual(
                yp,
                t=t,
                Fk_val=None,
                step_size=h_val,
                prev_state=prev_phys,
            )
            for qs in q_slices:
                out[qs] = -c_res[qs]
        return out

    def _physical_jacobian(t, yp, *, prev_state=None, h_val=None):
        key = None if h_val is None else float(h_val)
        if smooth_rhs_is_affine and key in _jac_phys_cache:
            return _jac_phys_cache[key]

        if rhs_jac is not None:
            J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
        else:
            J_s = _fd_smooth_jac(t, yp)
        J_s = _dense_or_sparse(J_s)
        if not sp.issparse(J_s):
            J_s = sp.csr_matrix(J_s)
        else:
            J_s = J_s.tocsr()

        if alg_proj is not None:
            prev_phys = None if prev_state is None else np.asarray(prev_state[:n_phys], dtype=float)
            patch = alg_proj.build_constraint_patch(
                yp,
                n_phys,
                t=t,
                Fk_val=None,
                step_size=h_val,
                prev_state=prev_phys,
            ).tocsr()
            J_s = J_s.tolil()
            for qs in q_slices:
                J_s[qs, :] = (-patch[qs, :]).tolil()
            J_s = J_s.tocsr()

        _jac_phys_cache[key] = J_s
        return J_s

    def _reaction_response_matrix(t, yp, *, prev_state=None, h_val=None):
        h_eff = 1.0 if h_val is None or h_val <= 0.0 else float(h_val)
        key = h_eff
        if smooth_rhs_is_affine and key in _response_cache:
            return _response_cache[key]

        J_phys = _physical_jacobian(t, yp, prev_state=prev_state, h_val=h_eff)
        K = (A_phys - h_eff * J_phys) if not sp.issparse(A_phys) else (A_phys - h_eff * J_phys)
        if sp.issparse(K):
            K_csc = K.tocsc()
            rhs_mat = (h_eff * _B_csr).toarray()
            try:
                lu = spla.splu(K_csc)
            except RuntimeError:
                reg = 1.0e-12 * max(1.0, spla.norm(K_csc))
                lu = spla.splu((K_csc + reg * sp.eye(n_phys, format="csc")).tocsc())
            G_phys_all = lu.solve(rhs_mat)
            _lu_cache[key] = lu
        else:
            rhs_mat = h_eff * np.asarray(B_mat, dtype=float)
            try:
                G_phys_all = np.linalg.solve(np.asarray(K, dtype=float), rhs_mat)
            except np.linalg.LinAlgError:
                reg = 1.0e-12 * max(1.0, np.linalg.norm(K, ord=2))
                G_phys_all = np.linalg.solve(
                    np.asarray(K, dtype=float) + reg * np.eye(n_phys), rhs_mat
                )
            _lu_cache[key] = np.asarray(K, dtype=float)
        _response_cache[key] = np.asarray(G_phys_all, dtype=float)
        return _response_cache[key]

    def _local_model_builder(
        *,
        t,
        current_state,
        candidate,
        active_blocks,
        Fk_val=None,
        prev_state=None,
        step_size=None,
        reaction_hint=None,
    ):
        y_bar_phys = np.asarray(candidate[:n_phys], dtype=float)
        y_cur_phys = np.asarray(current_state[:n_phys], dtype=float)
        prev_phys = None if prev_state is None else np.asarray(prev_state[:n_phys], dtype=float)
        G_phys_all = _reaction_response_matrix(
            t, y_bar_phys, prev_state=prev_phys, h_val=step_size
        )

        active_rows = []
        local_block_slices = []
        mu_all, beta_all = _vectorize_contact_params(y_cur_phys, t=t, Fk_val=Fk_val)
        mu_active = []
        alpha_active = []
        flat_pos = 0
        for block_idx in np.asarray(active_blocks, dtype=int):
            sl = norm_contacts[block_idx]["block_slice"]
            rows = np.arange(sl.start, sl.stop, dtype=int)
            active_rows.extend(rows.tolist())
            d = sl.stop - sl.start
            local_block_slices.append(slice(flat_pos, flat_pos + d))
            flat_pos += d
            mu_active.append(float(mu_all[block_idx]))
            alpha_active.append(float(mu_all[block_idx] - beta_all[block_idx]))

        active_rows = np.asarray(active_rows, dtype=int)
        G_state = np.asarray(G_phys_all[:, active_rows], dtype=float)

        U_active = U_contact_csr[active_rows, :]
        U_y = np.asarray(U_active.toarray(), dtype=float)

        if reaction_hint is None:
            reaction_hint = np.zeros(n_react, dtype=float)
        warm_start = np.asarray(reaction_hint, dtype=float)[active_rows]

        y_free_phys = None
        if prev_phys is not None and step_size is not None and step_size > 0:
            h_eff = float(step_size)
            _cached_lu = _lu_cache.get(h_eff)
            if _cached_lu is not None:
                f_prev = _physical_rhs(t, prev_phys, prev_state=prev_phys, h_val=h_eff)
                rhs_vec = h_eff * np.asarray(f_prev, dtype=float)
                if hasattr(_cached_lu, 'solve'):
                    y_free_phys = prev_phys + _cached_lu.solve(rhs_vec)
                else:
                    y_free_phys = prev_phys + np.linalg.solve(_cached_lu, rhs_vec)

        if y_free_phys is not None:
            u_free = np.asarray(U_contact_csr @ y_free_phys).ravel()[active_rows]
        else:
            u_free = np.asarray(U_contact_csr @ y_bar_phys).ravel()[active_rows]

        offset_active = None
        if _has_offset:
            offset_full = np.zeros(n_react, dtype=float)
            s0_arr = _eval_s0(get_s0, _s0_nargs, n_blocks, y_cur_phys,
                              t=t, Fk_val=Fk_val)
            for k_b, ci in enumerate(norm_contacts):
                sl_b = ci["block_slice"]
                offset_full[sl_b.start] = float(s0_arr[k_b])
                m_k = sl_b.stop - sl_b.start - 1
                if m_k > 0:
                    offset_full[sl_b.start + 1 : sl_b.stop] = _eval_w0(
                        get_w0, _w0_nargs, y_cur_phys, k_b, m_k,
                        t=t, Fk_val=Fk_val,
                    )
            offset_active = offset_full[active_rows]

        return {
            "u_free": u_free,
            "U_y": U_y,
            "G_state": G_state,
            "mu": np.asarray(mu_active, dtype=float),
            "alpha": np.asarray(alpha_active, dtype=float),
            "block_slices": local_block_slices,
            "warm_start": warm_start,
            "offset": offset_active,
            "y_free": y_free_phys,
        }

    proj = _DeSaxceConeProjection(
        n_phys=n_phys,
        n_react=n_react,
        block_slices=[ci["block_slice"] for ci in norm_contacts],
        gap=gap_aug,
        local_model_builder=_local_model_builder,
        reaction_offset=None,
        gap_tol=gap_tol,
        inactive_handling=inactive_handling,
        prox_rho=prox_rho,
        prox_rho_scale=prox_rho_scale,
        prox_rho_min=prox_rho_min,
        prox_rho_max=prox_rho_max,
        vi_maxit=vi_maxit,
        vi_tol=vi_tol,
        tc_tol=tc_tol,
        component_slices=cs_phys,
    )

    vel_dof_map = []
    for ci in norm_contacts:
        block_rows = np.arange(ci["block_slice"].start, ci["block_slice"].stop, dtype=int)
        cols = U_contact_csr[block_rows, :].indices
        vel_dof_map.append(np.unique(cols))
    proj._velocity_dof_map = vel_dof_map
    proj._vectorize_contact_params = _vectorize_contact_params
    proj._n_phys = n_phys

    def rhs_aug(t, y, *extra):
        prev_state, _, h_val = _parse_prev_and_h(extra, y.shape)
        yp = np.asarray(y[:n_phys], dtype=float)
        return _physical_rhs(t, yp, prev_state=prev_state, h_val=h_val)

    def jac_aug(t, y, *extra):
        prev_state, _, h_val = _parse_prev_and_h(extra, y.shape)
        yp = np.asarray(y[:n_phys], dtype=float)
        top_left = _physical_jacobian(t, yp, prev_state=prev_state, h_val=h_val)
        if sp.issparse(A_phys):
            return top_left.tocsr() if sp.issparse(top_left) else sp.csr_matrix(top_left)
        return top_left.toarray() if sp.issparse(top_left) else np.asarray(top_left, dtype=float)

    def _reaction_history(states, times):
        states = np.asarray(states, dtype=float)
        times = np.asarray(times, dtype=float)
        n_hist = states.shape[0]
        out = np.zeros((n_hist, n_react), dtype=float)
        for i in range(n_hist):
            prev_state = None if i == 0 else np.asarray(states[i - 1], dtype=float)
            h_step = None if i == 0 else float(times[i] - times[i - 1])
            out[i, :] = _reaction_from_step(
                np.asarray(states[i], dtype=float),
                prev_state=prev_state,
                t=float(times[i]),
                h_val=h_step,
            )
        return out

    _B_dense = np.asarray(_B_csr.toarray(), dtype=float)

    def _reaction_from_step(state, *, prev_state=None, t=None, h_val=None):
        if prev_state is None or h_val is None or h_val <= 0.0:
            return np.zeros(n_react, dtype=float)
        state = np.asarray(state, dtype=float)
        prev_state = np.asarray(prev_state, dtype=float)
        drive = np.asarray(A_phys @ ((state - prev_state) / h_val)).ravel()
        smooth = _physical_rhs(t, state, prev_state=prev_state, h_val=h_val)
        rhs_contact = drive - smooth
        if n_react == 0:
            return np.zeros(0, dtype=float)
        sol, *_ = np.linalg.lstsq(_B_dense, rhs_contact, rcond=None)
        return np.asarray(sol, dtype=float).ravel()

    cs = ContactSystem(
        A=A_phys,
        rhs=rhs_aug,
        y0=y0,
        projection=proj,
        component_slices=cs_phys,
        integrator_opts={"pass_prev_state": True, "pass_step_size": True},
        n_phys=n_phys,
        B=B_mat,
        rhs_jac=jac_aug,
    )
    cs.reaction_history = _reaction_history
    cs.reaction_from_step = _reaction_from_step
    cs.n_react = n_react
    return cs


def build_dynamic_desaxce_projected_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    component_slices=None,
    gap_extract=None,
    vel_extract=None,
    constraints=None,
    rhs_jac=None,
    gap_tol=0.0,
    inactive_handling="gap",
    prox_rho="auto",
    prox_rho_scale=1.0,
    prox_rho_min=1.0e-8,
    prox_rho_max=1.0e8,
    vi_maxit=30,
    vi_tol=1.0e-10,
    tc_tol=1.0e-10,
):
    r"""Build a De Saxce contact system with an explicit post-step projection stage.

    This is the closest in-package analogue to the dissertation-style
    projected RK structure: the smooth physical step is solved first and the
    nonsmooth De Saxce correction is then applied explicitly as an end-of-step
    projection, rather than being embedded inside the Newton residual.

    The returned :class:`~solve_nivp.contact.ContactSystem` remains physical-state
    only. Its usual ``projection`` field is replaced by an identity map for the
    nonlinear solve, while the true contact projector is threaded through the
    integrator via ``integrator_opts['post_step_projection']``.
    """
    cs = build_dynamic_desaxce_contact(
        A=A,
        rhs_smooth=rhs_smooth,
        y0=y0,
        contacts=contacts,
        gap_func=gap_func,
        B=B,
        component_slices=component_slices,
        gap_extract=gap_extract,
        vel_extract=vel_extract,
        constraints=constraints,
        rhs_jac=rhs_jac,
        gap_tol=gap_tol,
        inactive_handling=inactive_handling,
        prox_rho=prox_rho,
        prox_rho_scale=prox_rho_scale,
        prox_rho_min=prox_rho_min,
        prox_rho_max=prox_rho_max,
        vi_maxit=vi_maxit,
        vi_tol=vi_tol,
        tc_tol=tc_tol,
    )
    step_projection = cs.projection
    cs.step_projection = step_projection
    cs.projection = IdentityProjection(component_slices=cs.component_slices)
    cs.integrator_opts = dict(cs.integrator_opts)
    cs.integrator_opts["post_step_projection"] = step_projection
    cs.integrator_opts.setdefault("post_step_rhok", 1.0)
    return cs


def _desaxce_block_residual_and_jac(u_blk, r_blk, mu, alpha, rho, offset=None, delassus=None):
    """One-block De Saxce natural-map residual and generalized Jacobian.

    Computes the Euclidean De Saxce natural-map residual:

        phi(r, u) = r_eff - Proj_{K_mu}(r_eff - rho * u_hat(u))

    where ``r_eff = r + offset`` (prestress shift) and
    ``u_hat(u) = (u_N + alpha * ||u_T||, u_T)`` is the De Saxce augmented
    velocity.  The projection is the *Euclidean* projection onto the Coulomb cone
    K_mu = {(s, w) : ||w|| <= mu * s}.

    Note on self-dual rescaling
    ---------------------------
    A diagonal scaling S = diag(1, 1/mu) maps K_mu to the Lorentz cone K_1, but
    S is *not* an orthogonal transformation.  As a consequence the identity
    S @ Proj_{K_mu}(z) == Proj_{K_1}(S @ z) fails for mu != 1 (it holds only
    for orthogonal S).  Therefore projecting in rescaled coordinates onto K_1
    implements a different (S-weighted) contact law that has different zeros from
    the Euclidean De Saxce natural map.  The direct Euclidean path below is the
    only correct implementation.
    """
    u_blk = np.asarray(u_blk, dtype=float)
    r_blk = np.asarray(r_blk, dtype=float)
    d = r_blk.size

    r_eff = r_blk if offset is None else r_blk + np.asarray(offset, dtype=float)

    rho_vec = np.atleast_1d(np.asarray(rho, dtype=float))
    if rho_vec.size == 1:
        rho_vec = np.full(d, float(rho_vec[0]), dtype=float)

    u_hat, D_uhat = _DeSaxceConeProjection._uhat_and_jac(
        u_blk, np.array([alpha], dtype=float), [slice(0, d)]
    )
    z = r_eff - rho_vec * u_hat
    proj_z, J_proj = MuScaledSOCProjection._proj_mu_scaled_soc(
        z, float(mu), return_jacobian=True
    )
    phi = r_eff - proj_z
    dphi_du = J_proj @ (np.diag(rho_vec) @ D_uhat)
    if delassus is not None:
        W = np.asarray(delassus, dtype=float)
        if W.ndim == 1:
            W = np.diag(W)
        rhoD_W = np.diag(rho_vec) @ D_uhat @ W
        dphi_dr = np.eye(d, dtype=float) - J_proj @ (np.eye(d, dtype=float) - rhoD_W)
    else:
        dphi_dr = np.eye(d, dtype=float) - J_proj
    return phi, dphi_du, dphi_dr


def build_dynamic_desaxce_residual_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    component_slices=None,
    gap_extract=None,
    vel_extract=None,
    constraints=None,
    rhs_jac=None,
    gap_tol=0.0,
    contact_rho=1.0,
    inactive_handling="natural_map",
    reaction_units="force",
    normal_mode="velocity",
    normal_gap_scale=1.0,
    get_s0=None,
    get_w0=None,
    smooth_rhs_is_affine=False,
):
    r"""Build a full-state De Saxce natural-map contact system.

    This backend augments the physical state with contact reactions and writes
    the De Saxce cone law directly into the residual:

    ``phi(r, u) = r - Proj_{K_mu}(r - rho * u_hat(u)) = 0``.

    Unlike :func:`build_dynamic_desaxce_contact`, this path does not rely on a
    reduced-space projection. It is intended for large FEM cases where the full
    Newton residual is the more robust fit for the existing solver stack.

    Parameters
    ----------
    reaction_units : {"force", "impulse"}, default "force"
        Units carried by the reaction unknowns ``r``.

        ``"force"`` (default): ``r`` is the contact force/stress.  The
        physical rows couple through ``B r`` in the continuous-time RHS
        ``A ẏ = f(y) + B r``.  Backward Euler integrates this to
        ``A Δy/h = f + B r``, so ``r`` appears scaled by ``h`` in the
        momentum balance.  The proximal parameter ``contact_rho`` has
        units [force/velocity]; for a well-conditioned natural map it
        should scale with the contact compliance (~1/stiffness), **not**
        the timestep.

        ``"impulse"``: ``r_imp = h * r_force`` is the contact impulse.
        The physical coupling is ``B r_imp / h = B r_force`` so the
        momentum equation is unchanged, but ``r`` is now h-scaled.  The
        proximal parameter ``contact_rho`` has units
        [impulse/velocity] ≈ [mass] and is **h-independent**, giving
        consistent Newton conditioning across varying timestep sizes.
        Recommended for adaptive time-stepping or impact-dominated
        problems.

    contact_rho : float or "auto", default 1.0
        Proximal parameter ``ρ`` in ``φ(r, u) = r - P_{K_μ}(r - ρ û)``.

        With ``reaction_units="force"``, ``ρ`` has units [force/velocity].
        A reasonable heuristic is ``ρ ≈ 1 / (contact_stiffness * h)`` so
        that the proximal point shift ``ρ û`` is comparable in magnitude to
        ``r``.

        With ``reaction_units="impulse"``, ``ρ`` has units
        [impulse/velocity] ≈ [mass] and should be chosen as a
        characteristic mass of the contacting bodies — h-independent.

        ``"auto"`` (experimental): estimates ``ρ`` per block from the
        diagonal of ``A`` divided by the number of contact blocks.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size
    gap_tol = float(gap_tol)

    if gap_extract is not None:
        gap_extract = _dense_or_sparse(gap_extract)
        if gap_extract.shape[1] != n_phys:
            raise ValueError(
                f"gap_extract has {gap_extract.shape[1]} columns but n_phys = {n_phys}"
            )
    if vel_extract is not None:
        vel_extract = _dense_or_sparse(vel_extract)
        if vel_extract.shape[1] != n_phys:
            raise ValueError(
                f"vel_extract has {vel_extract.shape[1]} columns but n_phys = {n_phys}"
            )
    if gap_func is None and gap_extract is None:
        raise ValueError("gap_func must be provided when gap_extract is None")
    inactive_handling = str(inactive_handling).strip().lower().replace("-", "_")
    if inactive_handling not in {"hard_zero", "natural_map"}:
        raise ValueError(
            "inactive_handling must be 'hard_zero' or 'natural_map' "
            f"(got {inactive_handling!r})"
        )
    normal_mode = str(normal_mode).strip().lower().replace("-", "_")
    if normal_mode not in {"velocity", "gap_be", "gap_plus_velocity_be"}:
        raise ValueError(
            "normal_mode must be 'velocity', 'gap_be', or 'gap_plus_velocity_be' "
            f"(got {normal_mode!r})"
        )
    normal_gap_scale = float(normal_gap_scale)
    if not np.isfinite(normal_gap_scale) or normal_gap_scale < 0.0:
        raise ValueError(
            "normal_gap_scale must be a finite nonnegative scalar "
            f"(got {normal_gap_scale!r})"
        )
    reaction_units = str(reaction_units).strip().lower()
    if reaction_units not in {"force", "impulse"}:
        raise ValueError(
            "reaction_units must be 'force' or 'impulse' "
            f"(got {reaction_units!r})"
        )
    # Resolve contact_rho: scalar, callable, or "auto".
    _rho_auto = isinstance(contact_rho, str) and contact_rho.strip().lower() == "auto"
    if _rho_auto:
        # Estimate rho from the diagonal of A (characteristic mass/damping scale).
        # Deferred until n_react is known; set placeholder now.
        _rho_scalar = None
    elif callable(contact_rho):
        _rho_scalar = None  # evaluated per-call below
    else:
        _rho_scalar = float(contact_rho)
    rho_nargs = _count_required_args(contact_rho) if callable(contact_rho) else None

    norm_contacts = []
    reaction_extract_rows = []
    reaction_idx = 0
    for c in contacts:
        e_val = float(c.get("e", 0.0))
        if abs(e_val) > 1.0e-14:
            raise ValueError(
                "build_dynamic_desaxce_residual_contact only supports zero restitution (e = 0)"
            )

        v_n = int(c["vel_normal_idx"])
        v_t = list(np.atleast_1d(c.get("vel_tangential_idx", [])).astype(int))

        mu_val = c.get("mu", 0.0)
        if callable(mu_val):
            get_mu = mu_val
        else:
            mu_const = float(mu_val)

            def get_mu(y, t=None, Fk_val=None, _m=mu_const):  # noqa: E306
                return _m

        beta_val = c.get("beta", 0.0)
        if callable(beta_val):
            get_beta = beta_val
        else:
            beta_const = float(beta_val)

            def get_beta(y, t=None, Fk_val=None, _b=beta_const):  # noqa: E306
                return _b

        block_slice = slice(reaction_idx, reaction_idx + 1 + len(v_t))
        reaction_extract_rows.extend([v_n] + v_t)
        norm_contacts.append(
            {
                "vN": v_n,
                "vT": v_t,
                "block_slice": block_slice,
                "get_mu": get_mu,
                "mu_nargs": _count_required_args(get_mu),
                "get_beta": get_beta,
                "beta_nargs": _count_required_args(get_beta),
            }
        )
        reaction_idx += 1 + len(v_t)

    n_react = reaction_idx
    n_aug = n_phys + n_react

    # Diagonal of A for Schur/Delassus estimates (auto-rho + Jacobian
    # coupling correction that prevents rank deficiency on cone boundary).
    if sp.issparse(A):
        _A_diag_rho = np.abs(np.asarray(A.diagonal()).ravel())
    else:
        _A_diag_rho = np.abs(np.diag(np.asarray(A, dtype=float)))
    _pos_rho = _A_diag_rho > 0
    _A_diag_rho = np.where(
        _pos_rho, _A_diag_rho,
        (_A_diag_rho[_pos_rho].min() if _pos_rho.any() else 1.0),
    )

    _rho_schur_base = None
    _rho_h_cell = [1.0]

    if B is None and vel_extract is not None:
        if sp.issparse(vel_extract):
            B_mat = vel_extract[reaction_extract_rows, :].T.tocsr()
        else:
            B_mat = np.asarray(vel_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None and gap_extract is not None:
        if sp.issparse(gap_extract):
            B_mat = gap_extract[reaction_extract_rows, :].T.tocsr()
        else:
            B_mat = np.asarray(gap_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None:
        B_mat = np.zeros((n_phys, n_react), dtype=float)
        col = 0
        for ci in norm_contacts:
            B_mat[ci["vN"], col] = 1.0
            col += 1
            for vt in ci["vT"]:
                B_mat[vt, col] = 1.0
                col += 1
    else:
        B_mat = _dense_or_sparse(B)
        if B_mat.shape != (n_phys, n_react):
            raise ValueError(
                f"B shape {B_mat.shape} doesn't match (n_phys={n_phys}, n_react={n_react})"
            )

    # Per-component Schur diagonal: base_i = Σ_j B_ji² / A_jj.
    # Used for auto-rho AND diagonal Delassus coupling correction in jac_aug
    # (approximates the condensed W = U (A/h − J)⁻¹ B to prevent rank
    # deficiency of I − J_proj on the 2D cone boundary).
    if sp.issparse(B_mat):
        _B_dense_rho = B_mat.toarray()
    else:
        _B_dense_rho = np.asarray(B_mat, dtype=float)
    _rho_schur_per_comp = np.ones(n_react, dtype=float)
    for _k, _ci in enumerate(norm_contacts):
        _sl = _ci["block_slice"]
        for _col_idx in range(_sl.start, _sl.stop):
            _b_col = _B_dense_rho[:, _col_idx]
            _base = float(np.sum(_b_col ** 2 / _A_diag_rho))
            _rho_schur_per_comp[_col_idx] = max(_base, 1.0e-30)

    if sp.issparse(A):
        A_aug = sp.block_diag([A.tocsr(), sp.csr_matrix((n_react, n_react))], format="csr")
    else:
        A_aug = np.zeros((n_aug, n_aug), dtype=float)
        A_aug[:n_phys, :n_phys] = np.asarray(A, dtype=float)

    y0_aug = np.zeros(n_aug, dtype=float)
    y0_aug[:n_phys] = y0

    if gap_func is not None:

        def gap_aug(y, t=None):
            return np.atleast_1d(gap_func(np.asarray(y[:n_phys], dtype=float), t))

    else:
        normal_rows = [ci["vN"] for ci in norm_contacts]

        def gap_aug(y, t=None):
            vals = gap_extract @ np.asarray(y[:n_phys], dtype=float)
            vals = np.asarray(vals).ravel()
            return vals[normal_rows]

    gap_contact_rows_csr = None
    if normal_mode in {"gap_be", "gap_plus_velocity_be"}:
        if gap_extract is None:
            raise ValueError(
                f"normal_mode={normal_mode!r} requires matrix gap_extract so the "
                "gap Jacobian is available."
            )
        if sp.issparse(gap_extract):
            gap_contact_rows_csr = gap_extract[normal_rows, :].tocsr()
        else:
            gap_contact_rows_csr = sp.csr_matrix(
                np.asarray(gap_extract[normal_rows, :], dtype=float)
            )

    if vel_extract is not None:
        U_contact = vel_extract[reaction_extract_rows, :]
        U_contact_csr = (
            U_contact.tocsr() if sp.issparse(U_contact) else sp.csr_matrix(U_contact)
        )
    else:
        vel_indices = np.asarray(reaction_extract_rows, dtype=int)
        U_contact_csr = sp.csr_matrix(
            (
                np.ones(n_react, dtype=float),
                (np.arange(n_react, dtype=int), vel_indices),
            ),
            shape=(n_react, n_phys),
        )

    alg_proj = None
    q_slices = []
    if constraints is not None:
        alg_proj = AlgebraicConstraintProjection(constraints=constraints)
        q_slices = list(alg_proj.constraint_q_slices)

    if component_slices is not None:
        cs_aug = []
        any_array = False
        for cs_item in component_slices:
            if isinstance(cs_item, slice):
                cs_aug.append(cs_item)
            else:
                cs_aug.append(np.asarray(cs_item, dtype=int))
                any_array = True
        cs_aug.append(
            np.arange(n_phys, n_aug, dtype=int) if any_array else slice(n_phys, n_aug)
        )
    else:
        vel_set = set(reaction_extract_rows)
        vel_idx = np.array(sorted(vel_set), dtype=int)
        other_idx = np.array(sorted(set(range(n_phys)) - vel_set), dtype=int)
        react_idx = np.arange(n_phys, n_aug, dtype=int)
        cs_aug = []
        if vel_idx.size > 0:
            cs_aug.append(vel_idx)
        if other_idx.size > 0:
            cs_aug.append(other_idx)
        cs_aug.append(react_idx)

    proj = IdentityProjection(component_slices=cs_aug)

    _has_offset = get_s0 is not None or get_w0 is not None
    s0_nargs = _count_required_args(get_s0) if callable(get_s0) else None
    w0_nargs = _count_required_args(get_w0) if callable(get_w0) else None
    n_blocks = len(norm_contacts)

    def _assemble_offset_vector(y_full, *, t=None, Fk_val=None):
        offset_vec = np.zeros(n_react, dtype=float)
        if not _has_offset:
            return offset_vec
        yp = y_full[:n_phys]
        s0_arr = _eval_s0(get_s0, s0_nargs, n_blocks, yp, t=t, Fk_val=Fk_val)
        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            offset_vec[sl.start] = float(s0_arr[k])
            m_k = sl.stop - sl.start - 1
            if m_k > 0:
                offset_vec[sl.start + 1 : sl.stop] = _eval_w0(
                    get_w0, w0_nargs, yp, k, m_k, t=t, Fk_val=Fk_val
                )
        return offset_vec

    _B_csr = B_mat.tocsr() if sp.issparse(B_mat) else sp.csr_matrix(B_mat)
    _jac_phys_cache = {}
    _smooth_rhs_affine = [None]
    _smooth_rhs_linear = [None]
    _smooth_rhs_cache_t = [None]

    def _vectorize_contact_params(y_state, *, t=None, Fk_val=None):
        mu = np.empty(len(norm_contacts), dtype=float)
        beta = np.empty(len(norm_contacts), dtype=float)
        for k, ci in enumerate(norm_contacts):
            mu[k] = float(
                _call_state_time_fk(ci["get_mu"], ci["mu_nargs"], y_state, t, Fk_val)
            )
            beta[k] = float(
                _call_state_time_fk(
                    ci["get_beta"], ci["beta_nargs"], y_state, t, Fk_val
                )
            )
        if np.any(beta < -1.0e-14):
            raise ValueError("beta must be nonnegative in build_dynamic_desaxce_residual_contact")
        if np.any(beta > mu + 1.0e-14):
            raise ValueError("beta must satisfy beta <= mu in build_dynamic_desaxce_residual_contact")
        return mu, np.clip(beta, 0.0, mu)

    def _vectorize_rho(y_state, *, t=None, Fk_val=None):
        """Return per-block rho arrays.

        When ``contact_rho='auto'``, returns a list of per-component vectors
        (one per block) with separate normal/tangential scaling:
        ``rho_N = 1/(h²·base_N)`` and ``rho_T = 1/(h·base_T)`` for force
        units, mirroring the NCP normal_r/friction_r split.

        For scalar or callable rho, returns a list of scalar-broadcast arrays.
        """
        n_blocks = len(norm_contacts)
        if _rho_auto:
            h = max(_rho_h_cell[0], 1.0e-30)
            rho_list = []
            for _ci in norm_contacts:
                _sl = _ci["block_slice"]
                d = _sl.stop - _sl.start
                rho_blk = np.empty(d, dtype=float)
                base_n = _rho_schur_per_comp[_sl.start]
                if reaction_units == "impulse":
                    rho_blk[0] = 1.0 / max(h * base_n, 1.0e-30)
                else:
                    rho_blk[0] = 1.0 / max(h ** 2 * base_n, 1.0e-30)
                for j in range(1, d):
                    base_t = _rho_schur_per_comp[_sl.start + j]
                    if reaction_units == "impulse":
                        rho_blk[j] = 1.0 / max(base_t, 1.0e-30)
                    else:
                        rho_blk[j] = 1.0 / max(h * base_t, 1.0e-30)
                rho_list.append(rho_blk)
            return rho_list
        if callable(contact_rho):
            rho_val = _call_state_time_fk(contact_rho, rho_nargs, y_state, t, Fk_val)
        elif _rho_scalar is not None:
            rho_val = _rho_scalar
        else:
            rho_val = contact_rho
        rho_arr = np.atleast_1d(np.asarray(rho_val, dtype=float))
        if rho_arr.size == 1:
            return [np.full(ci["block_slice"].stop - ci["block_slice"].start,
                            float(rho_arr[0]), dtype=float) for ci in norm_contacts]
        if rho_arr.size == n_blocks:
            return [np.full(ci["block_slice"].stop - ci["block_slice"].start,
                            float(rho_arr[k]), dtype=float) for k, ci in enumerate(norm_contacts)]
        raise ValueError(
            f"contact_rho must return one value or {n_blocks} values (got {rho_arr.size})"
        )

    def _fd_smooth_jac(t, yp):
        f0 = _call_with_time_state_fk(rhs_smooth, t, yp, None)
        eps_base = 1.0e-7
        h_vec = eps_base * np.maximum(np.abs(yp), 1.0)
        J = np.empty((n_phys, n_phys), dtype=float)
        for j in range(n_phys):
            yp_pert = yp.copy()
            yp_pert[j] += h_vec[j]
            fp = _call_with_time_state_fk(rhs_smooth, t, yp_pert, None)
            J[:, j] = (fp - f0) / h_vec[j]
        return J

    def _physical_rhs(t, yp, *, prev_state=None, h_val=None):
        out = np.zeros(n_phys, dtype=float)
        if smooth_rhs_is_affine and rhs_jac is not None:
            t_key = None if t is None else float(t)
            if _smooth_rhs_linear[0] is None or _smooth_rhs_cache_t[0] != t_key:
                J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
                J_s = _dense_or_sparse(J_s)
                _smooth_rhs_linear[0] = (
                    J_s.tocsr() if sp.issparse(J_s) else sp.csr_matrix(J_s)
                )
                _smooth_rhs_affine[0] = np.asarray(
                    _call_with_time_state_fk(rhs_smooth, t, np.zeros(n_phys), None)
                ).ravel()
                _smooth_rhs_cache_t[0] = t_key
            out[:] = np.asarray(_smooth_rhs_linear[0] @ yp).ravel() + _smooth_rhs_affine[0]
        else:
            out[:] = np.asarray(_call_with_time_state_fk(rhs_smooth, t, yp, None)).ravel()

        if alg_proj is not None:
            prev_phys = None if prev_state is None else np.asarray(prev_state[:n_phys], dtype=float)
            c_res = alg_proj.constraint_residual(
                yp,
                t=t,
                Fk_val=None,
                step_size=h_val,
                prev_state=prev_phys,
            )
            for qs in q_slices:
                out[qs] = -c_res[qs]
        return out

    def _physical_jacobian(t, yp, *, prev_state=None, h_val=None):
        key = None if h_val is None else float(h_val)
        if smooth_rhs_is_affine and key in _jac_phys_cache:
            return _jac_phys_cache[key]

        if rhs_jac is not None:
            J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
        else:
            J_s = _fd_smooth_jac(t, yp)
        J_s = _dense_or_sparse(J_s)
        if not sp.issparse(J_s):
            J_s = sp.csr_matrix(J_s)
        else:
            J_s = J_s.tocsr()

        if alg_proj is not None:
            prev_phys = None if prev_state is None else np.asarray(prev_state[:n_phys], dtype=float)
            patch = alg_proj.build_constraint_patch(
                yp,
                n_phys,
                t=t,
                Fk_val=None,
                step_size=h_val,
                prev_state=prev_phys,
            ).tocsr()
            J_s = J_s.tolil()
            for qs in q_slices:
                J_s[qs, :] = (-patch[qs, :]).tolil()
            J_s = J_s.tocsr()

        _jac_phys_cache[key] = J_s
        return J_s

    def rhs_aug(t, y, *extra):
        prev_state, Fk_val, h_val = _parse_prev_and_h(extra, y.shape)
        yp = np.asarray(y[:n_phys], dtype=float)
        r = np.asarray(y[n_phys:], dtype=float)
        out = np.zeros(n_aug, dtype=float)
        out[:n_phys] = _physical_rhs(t, yp, prev_state=prev_state, h_val=h_val)

        # Impulse units: r carries h*force; divide by h to recover force for the
        # continuous-time coupling B*f_c = B*(r_imp/h).
        if reaction_units == "impulse" and h_val is not None and h_val > 0.0:
            out[:n_phys] += np.asarray(_B_csr @ (r / float(h_val))).ravel()
        else:
            out[:n_phys] += np.asarray(_B_csr @ r).ravel()

        if _rho_auto and h_val is not None and h_val > 0.0:
            _rho_h_cell[0] = h_val
        u_rel = np.asarray(U_contact_csr @ yp).ravel()
        gaps = np.atleast_1d(gap_aug(y, t))
        mu_all, beta_all = _vectorize_contact_params(yp, t=t, Fk_val=Fk_val)
        rho_all = _vectorize_rho(yp, t=t, Fk_val=Fk_val)
        offset_vec = _assemble_offset_vector(y, t=t, Fk_val=Fk_val)

        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            r_blk = r[sl]
            offset_blk = offset_vec[sl] if _has_offset else None
            if inactive_handling == "hard_zero" and float(gaps[k]) > gap_tol:
                # Open contact: enforce zero reaction (r = 0), not r = -offset.
                phi = r_blk.copy()
            else:
                u_blk = np.asarray(u_rel[sl], dtype=float).copy()
                if normal_mode in {"gap_be", "gap_plus_velocity_be"} and h_val is not None and h_val > 0.0:
                    gap_rate_k = normal_gap_scale * float(gaps[k]) / float(h_val)
                    if normal_mode == "gap_plus_velocity_be":
                        u_blk[0] = float(u_blk[0]) + gap_rate_k
                    else:
                        u_blk[0] = gap_rate_k
                phi, _, _ = _desaxce_block_residual_and_jac(
                    u_blk,
                    r_blk,
                    mu_all[k],
                    mu_all[k] - beta_all[k],
                    rho_all[k],
                    offset=offset_blk,
                )
            out[n_phys + sl.start : n_phys + sl.stop] = -phi
        return out

    def jac_aug(t, y, *extra):
        prev_state, Fk_val, h_val = _parse_prev_and_h(extra, y.shape)
        yp = np.asarray(y[:n_phys], dtype=float)
        top_left = _physical_jacobian(t, yp, prev_state=prev_state, h_val=h_val)
        top_left = top_left.tocsr() if sp.issparse(top_left) else sp.csr_matrix(top_left)

        # Scale the B coupling block for impulse units (B/h instead of B).
        if reaction_units == "impulse" and h_val is not None and h_val > 0.0:
            top_right = _B_csr.multiply(1.0 / float(h_val)).tocsr()
        else:
            top_right = _B_csr.tocsr()

        if _rho_auto and h_val is not None and h_val > 0.0:
            _rho_h_cell[0] = h_val
        u_rel = np.asarray(U_contact_csr @ yp).ravel()
        gaps = np.atleast_1d(gap_aug(y, t))
        r = np.asarray(y[n_phys:], dtype=float)
        mu_all, beta_all = _vectorize_contact_params(yp, t=t, Fk_val=Fk_val)
        rho_all = _vectorize_rho(yp, t=t, Fk_val=Fk_val)
        offset_vec = _assemble_offset_vector(y, t=t, Fk_val=Fk_val)

        bl_parts = []
        br_dense = np.zeros((n_react, n_react), dtype=float)
        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            d = sl.stop - sl.start
            U_blk = U_contact_csr[sl, :].toarray()
            offset_blk = offset_vec[sl] if _has_offset else None
            if inactive_handling == "hard_zero" and float(gaps[k]) > gap_tol:
                dphi_du = np.zeros((d, d), dtype=float)
                dphi_dr = np.eye(d, dtype=float)
            else:
                u_blk = np.asarray(u_rel[sl], dtype=float).copy()
                if normal_mode in {"gap_be", "gap_plus_velocity_be"} and h_val is not None and h_val > 0.0:
                    gap_rate_k = normal_gap_scale * float(gaps[k]) / float(h_val)
                    if normal_mode == "gap_plus_velocity_be":
                        u_blk[0] = float(u_blk[0]) + gap_rate_k
                    else:
                        u_blk[0] = gap_rate_k
                    U_blk = U_blk.copy()
                    gap_row = (
                        normal_gap_scale
                        * gap_contact_rows_csr.getrow(k).toarray().ravel()
                        / float(h_val)
                    )
                    if normal_mode == "gap_plus_velocity_be":
                        U_blk[0, :] = U_blk[0, :] + gap_row
                    else:
                        U_blk[0, :] = gap_row
                phi, dphi_du, dphi_dr = _desaxce_block_residual_and_jac(
                    u_blk,
                    r[sl],
                    mu_all[k],
                    mu_all[k] - beta_all[k],
                    rho_all[k],
                    offset=offset_blk,
                )
                del phi

            bl_parts.append(sp.csr_matrix(-(dphi_du @ U_blk)))
            br_dense[sl, sl] = -dphi_dr

        bottom_left = sp.vstack(bl_parts, format="csr") if bl_parts else sp.csr_matrix((0, n_phys))
        bottom_right = sp.csr_matrix(br_dense)
        return sp.bmat(
            [[top_left, top_right], [bottom_left, bottom_right]],
            format="csr",
        )

    return ContactSystem(
        A=A_aug,
        rhs=rhs_aug,
        y0=y0_aug,
        projection=proj,
        component_slices=cs_aug,
        integrator_opts={"pass_prev_state": True, "pass_step_size": True},
        n_phys=n_phys,
        B=B_mat,
        rhs_jac=jac_aug,
    )
