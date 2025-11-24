"""Projection operators for nonsmooth constraints.

Each projection supplies two methods:

``project(current_state, candidate, rhok=None, t=None, Fk_val=None)``
    Return the projected point (often applied to ``candidate = y - lam*F(y)``).

``tangent_cone(candidate, current_state, rhok=None, t=None, Fk_val=None)``
    Return an (n,n) generalized derivative (Clarke selection) of the projector
    used to assemble semismooth Newton Jacobians. Sparse CSR matrices are
    supported and encouraged for large problems.

The norm of the projection residual ``||y - project(y, y - lam F(y))||`` drives
nonlinear convergence tests in both VI and semismooth Newton solvers.
"""

import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod

try:
    from ._numba_accel import NUMBA_AVAILABLE as _NUMBA_OK, projD_optimized_nb, classify_regions_nb
except Exception:  # pragma: no cover
    _NUMBA_OK = False
    def projD_optimized_nb(v, z, friction_vals):
        raise RuntimeError("numba path unavailable")
    def classify_regions_nb(v_arr, zt_arr, tol):
        raise RuntimeError("numba path unavailable")


##############################################################################
# Utility functions (NumPy-only)
##############################################################################

def _safe_at_set(arr, idx, val):
    arr[idx] = val
    return arr

def _safe_at_set_vector(arr, idxs, vals):
    arr[idxs] = vals
    return arr

def _to_numpy_if_needed(x):
    return x


##############################################################################
# Base Projection
##############################################################################
class Projection(ABC):
    """Abstract projection interface.

    Methods
    -------
    project(current_state, candidate, rhok=None, t=None, Fk_val=None) -> np.ndarray
        Project the candidate state given the current_state and optional parameters.
    tangent_cone(candidate, current_state, rhok=None, t=None, Fk_val=None)
        Return a generalized derivative (Clarke selection) of the projector at the point.
    """
    def __init__(self, component_slices=None):
        self.component_slices = component_slices if component_slices is not None else []

    @abstractmethod
    def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
        pass

    @abstractmethod
    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        pass

    # Optional batched APIs (default fallbacks preserve behavior)
    def project_batch(self, current_state, candidates, rhok=None, t=None, Fk_val=None):
        candidates = np.asarray(candidates)
        if candidates.ndim == 1:
            return self.project(current_state, candidates, rhok=rhok, t=t, Fk_val=Fk_val)
        out = np.empty_like(candidates)
        # Default: row-wise call (safe but not fast); subclasses may override
        for i in range(candidates.shape[0]):
            out[i] = self.project(current_state, candidates[i], rhok=rhok, t=t, Fk_val=Fk_val)
        return out


##############################################################################
# IdentityProjection
##############################################################################
class IdentityProjection(Projection):
    """Trivial projection that returns the candidate unchanged.

    tangent_cone returns the identity matrix of appropriate shape.
    """
    def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
        return candidate

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        n = candidate.shape[0]
        return np.eye(n)

    def project_batch(self, current_state, candidates, rhok=None, t=None, Fk_val=None):
        return np.asarray(candidates)


##############################################################################
# SignProjection
##############################################################################
class SignProjection(Projection):
    """
    Enforce s ∈ N_{[-1,1]}(w) via the resolvent:
        w := Proj_{[-1,1]}( w + tau * s ).
    At a fixed point this is equivalent to w ∈ sign(s).

    Jacobian (Clarke selection) for z = w + tau*s:
      if |z| < 1:  ∂w/∂w = 1,  ∂w/∂s = tau
      if |z| > 1:  ∂w/∂w = 0,  ∂w/∂s = 0
      if |z| ≈ 1:  use 0.5 and 0.5*tau (tie-break)
    """
    def __init__(self, y_indices, w_indices, tau=1.0, component_slices=None):
        super().__init__(component_slices=component_slices)
        self.y_indices = np.array(y_indices) if not np.isscalar(y_indices) else y_indices
        self.w_indices = np.array(w_indices) if not np.isscalar(w_indices) else w_indices
        self.tau = float(tau)

    def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
        new = candidate.copy()
        y = np.atleast_1d(new[self.y_indices])
        w = np.atleast_1d(new[self.w_indices])
        tau = self.tau if rhok is None else rhok
        z = w + tau * y
        w_new = np.clip(z, -1.0, 1.0)
        w_new_arr = np.asarray(w_new)
        if w_new_arr.ndim == 0 or w_new_arr.size == 1:
            new[self.w_indices] = float(w_new_arr.reshape(-1)[0])
        else:
            new[self.w_indices] = w_new_arr
        return new

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        n = candidate.shape[0]
        y = np.atleast_1d(candidate[self.y_indices])
        w = np.atleast_1d(candidate[self.w_indices])
        tau = self.tau if rhok is None else rhok
        tau_arr = np.broadcast_to(tau, y.shape)
        z = w + tau_arr * y

        D = np.eye(n)

        tol = 1e-12 * (1.0 + np.abs(z))
        interior = (np.abs(z) < 1.0 - tol)
        exterior = (np.abs(z) > 1.0 + tol)
        boundary = ~(interior | exterior)

        w_idx = np.atleast_1d(self.w_indices)
        y_idx = np.atleast_1d(self.y_indices)

        # exterior (clamped): derivative 0 wrt w and s
        if np.any(exterior):
            D[w_idx[exterior], w_idx[exterior]] = 0.0  # ∂w/∂w
            # ∂w/∂s already 0

        # interior (free): ∂w/∂w = 1, ∂w/∂s = tau
        if np.any(interior):
            D[w_idx[interior], w_idx[interior]] = 1.0
            # add cross term
            for (j_w, j_s, tau_val) in zip(w_idx[interior], y_idx[interior], tau_arr[interior]):
                D[j_w, j_s] = float(tau_val)

        # boundary (kink): Clarke selection 0.5
        if np.any(boundary):
            D[w_idx[boundary], w_idx[boundary]] = 0.5
            for (j_w, j_s, tau_val) in zip(w_idx[boundary], y_idx[boundary], tau_arr[boundary]):
                D[j_w, j_s] = 0.5 * float(tau_val)

        return D

    def project_batch(self, current_state, candidates, rhok=None, t=None, Fk_val=None):
        C = np.asarray(candidates)
        if C.ndim == 1:
            return self.project(current_state, C, rhok=rhok, t=t, Fk_val=Fk_val)
        out = C.copy()
        y = np.atleast_2d(out[:, np.atleast_1d(self.y_indices)])
        w = np.atleast_2d(out[:, np.atleast_1d(self.w_indices)])
        tau = self.tau if rhok is None else rhok
        z = w + tau * y
        w_new = np.clip(z, -1.0, 1.0)
        out[:, np.atleast_1d(self.w_indices)] = w_new
        return out




##############################################################################
# CoulombProjection 
##############################################################################
class CoulombProjection(Projection):
    def __init__(self,
                 con_force_func,
                 rhok,
                 component_slices=None,
                 constraint_indices=None,
                 jac_func=None,
                 conf_jacobian_mode: str = 'full',
                 use_numba='auto',
                 **kwargs):  # kwargs for backward compatibility (jac_mode etc.)
        """Coulomb-like projection with optional analytical constraint Jacobian.

        For each constrained index ``i`` an auxiliary value
        ``z_tilde_i = |state[i]| - rhok_i * conf_i(state)`` is assembled and the
        pair ``(v_i, z_tilde_i)`` is projected onto a monotone cone with region
        specific 2x2 projector blocks. Only the first coordinate is retained
        (reduced representation: no explicit augmentation in the state vector).

        Parameters
        ----------
        con_force_func : callable
            Constraint force function ``conf(y[, t[, Fk_val]]) -> ndarray``.
        rhok : float | sequence
            Scalar or per-block scaling (broadcast via ``component_slices`` when
            iterable) controlling the subtraction ``rhok_i * conf_i``.
        component_slices : list[slice], optional
            Partition for broadcasting ``rhok``; also used when deriving default
            constraint indices if ``constraint_indices`` is omitted.
        constraint_indices : sequence[int], optional
            Explicit constrained coordinate indices; overrides slice-derived.
        jac_func : callable, optional
            Analytical Jacobian of ``con_force_func``. Signature mirrors
            ``con_force_func``; if omitted finite differences are used.
        conf_jacobian_mode : {'full','none'}, default 'full'
            Skip expensive Jacobian evaluation when 'none'.
        use_numba : {'auto', True, False}, default 'auto'
            Enable numba accelerated kernels when available.

        Notes
        -----
        * Projection residual norm is derived from first coordinate only.
        * Tangent cone returns CSR with modified rows at constrained indices.
        * Numerical Jacobian uses forward differences with uniform step.
        """
        super().__init__(component_slices)
        self.con_force_func = con_force_func
        self.rhok = rhok
        self.jac_func = jac_func
        self.conf_jacobian_mode = conf_jacobian_mode if conf_jacobian_mode in ('full', 'none') else 'full'
        if constraint_indices is not None:
            self.constraint_indices = np.array(constraint_indices)
        else:
            if self.component_slices:
                self.constraint_indices = np.concatenate(
                    [np.arange(sl.start, sl.stop) for sl in self.component_slices]
                )
            else:
                self.constraint_indices = np.array([])
        self.use_numba = use_numba if use_numba in ('auto', True, False) else 'auto'

    def _call_con_force(self, y, t=None, Fk_val=None):
        # Flexible call to support signatures: (y), (y,t), (y,t,Fk_val)
        try:
            return self.con_force_func(y, t, Fk_val)
        except TypeError:
            try:
                return self.con_force_func(y, t)
            except TypeError:
                return self.con_force_func(y)

    def _compute_jacobian(self, y, t=None, Fk_val=None):
        if self.jac_func is not None:
            try:
                J = self.jac_func(y, t, Fk_val)
            except TypeError:
                J = self.jac_func(y)
            return _to_numpy_if_needed(J)
        return self._numerical_jacobian(y, t=t, Fk_val=Fk_val)

    def _numerical_jacobian(self, y, eps=1e-8, t=None, Fk_val=None):
        y_np = _to_numpy_if_needed(y)
        n = len(y_np)
        J = np.zeros((n,n), dtype=y_np.dtype)
        f0 = _to_numpy_if_needed(self._call_con_force(y, t=t, Fk_val=Fk_val))
        for j in range(n):
            y_pert = y_np.copy()
            y_pert[j] += eps
            f_eps = _to_numpy_if_needed(self._call_con_force(y_pert, t=t, Fk_val=Fk_val))
            J[:, j] = (f_eps - f0)/eps
        return J

    @staticmethod
    def _gather_rhok_ci(rhok, ci, component_slices):
        """Return rhok values aligned to constrained indices ``ci`` without building a full array.

        Accepts:
        - scalar rhok -> broadcast scalar
        - array-like matching state length -> direct indexing by ci
        - array-like matching number of component_slices -> broadcast per-slice values onto ci
        """
        if rhok is None:
            return 1.0  # scalar broadcast
        if np.isscalar(rhok):
            return float(rhok)
        rhok_arr = np.asarray(rhok)
        # If length matches state length, we can index directly
        # Otherwise, try per-slice mapping
        if rhok_arr.ndim == 1:
            # Heuristic: when provided per-slice
            if (component_slices is not None
                and len(component_slices) > 0
                and rhok_arr.size == len(component_slices)):
                rh_ci = np.empty(ci.size, dtype=float)
                # Map each ci to its owning slice index and assign corresponding rhok
                # This loop is over number of slices (typically small) and only fills ci positions
                pos = 0
                for k, sl in enumerate(component_slices):
                    # intersect ci with slice range
                    start = sl.start if hasattr(sl, 'start') else sl[0]
                    stop = sl.stop if hasattr(sl, 'stop') else sl[-1] + 1
                    mask = (ci >= start) & (ci < stop)
                    count = int(np.count_nonzero(mask))
                    if count:
                        rh_ci[pos:pos+count] = float(rhok_arr[k])
                        pos += count
                if pos != ci.size:
                    # Fallback: direct gather with clipping to bounds if slices unconventional
                    rh_ci = rhok_arr[np.clip(ci, 0, rhok_arr.size-1)]
                return rh_ci
            # Direct indexing (assumes rhok provided per-state)
            if rhok_arr.size >= np.max(ci)+1:
                return rhok_arr[ci]
        # Fallback: treat as scalar 1.0 to preserve robustness
        return 1.0

    @staticmethod
    def _projD_optimized(v, z, friction_vals):
        v = np.asarray(v)
        z = np.asarray(z)
        out_v = np.empty_like(v)
        out_z = np.empty_like(z)
        mask_con0 = (friction_vals == 0)
        mask_non0 = ~mask_con0
        out_v[mask_con0] = v[mask_con0]
        out_z[mask_con0] = z[mask_con0]
        if np.any(mask_non0):
            v2 = v[mask_non0]; z2 = z[mask_non0]
            R1 = (np.abs(z2) <= v2)
            R2 = (np.abs(v2) <= -z2)
            R3 = (np.abs(z2) <= -v2)
            s1 = 0.5 * (v2 + z2)
            s2 = 0.5 * (-v2 + z2)
            dv = np.empty_like(v2); dz = np.empty_like(z2)
            dv[:] = v2; dz[:] = z2
            dv[R1] = s1[R1]; dz[R1] = s1[R1]
            dv[R3] = -s2[R3]; dz[R3] =  s2[R3]
            dv[R2] = 0.0;     dz[R2] = 0.0
            out_v[mask_non0] = dv
            out_z[mask_non0] = dz
        return out_v, out_z

    @staticmethod
    def _projD(y, con_force_func, state, rhok, constraint_indices, t=None, Fk_val=None, use_numba=False, component_slices=None):
        # Normalize indices and early-out when no constraints
        ci = np.asarray(constraint_indices)
        if ci.size == 0:
            return _to_numpy_if_needed(y)

        # Flexible call for con_force (avoid if early-out above)
        try:
            conf = con_force_func(state, t, Fk_val)
        except TypeError:
            try:
                conf = con_force_func(state, t)
            except TypeError:
                conf = con_force_func(state)

        second_column = _to_numpy_if_needed(conf).copy()
        # Use abs(state[ci]) instead of auxiliary state[ci+1]
        st_ci_abs = np.abs(_to_numpy_if_needed(state[ci]))
        # Support scalar, per-state, or per-slice rhok without building a full-length array
        rhok_ci = CoulombProjection._gather_rhok_ci(rhok, ci, component_slices)
        conf_ci = _to_numpy_if_needed(second_column[ci])
        newvals = st_ci_abs - (rhok_ci * conf_ci)
        second_column[ci] = newvals

        y_np = _to_numpy_if_needed(y)
        second_np = _to_numpy_if_needed(second_column)
        fv = _to_numpy_if_needed(conf)

        if use_numba and _NUMBA_OK:
            # Ensure contiguous float64 arrays for best Numba performance
            y_c = np.ascontiguousarray(y_np, dtype=np.float64)
            sc_c = np.ascontiguousarray(second_np, dtype=np.float64)
            fv_c = np.ascontiguousarray(fv, dtype=np.float64)
            v_proj, z_proj = projD_optimized_nb(y_c, sc_c, fv_c)
        else:
            v_proj, z_proj = CoulombProjection._projD_optimized(y_np, second_np, fv)
        # No augmented state: do not write to ci+1
        return v_proj

    def project(self, current_state, candidate, rhok, t=None, Fk_val=None):
        # Early-out when there are no constrained indices
        ci = np.asarray(getattr(self, 'constraint_indices', np.array([])))
        if ci.size == 0:
            return candidate
        # Accept scalar or array rhok; avoid constructing a full-sized vector per call
        rhok_eff = 1.0 if rhok is None else rhok
        # decide on numba usage here
        if isinstance(self.use_numba, str):
            use_nb = (_NUMBA_OK and self.use_numba == 'auto')
        else:
            use_nb = bool(self.use_numba) and _NUMBA_OK
        return CoulombProjection._projD(
            candidate, self.con_force_func, current_state, rhok_eff, self.constraint_indices,
            t=t, Fk_val=Fk_val, use_numba=use_nb, component_slices=self.component_slices
        )

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        """
        Generalized derivative (Clarke selection) of the projection (exact region tests).

        For each constrained index i, define z_tilde = |current_state[i]| - rhok[i] * conf_i(current_state).
        The projection of (v, z_tilde) onto the monotone cone uses regions with projector blocks P.
        Chain rule accounts for dependence of z_tilde on the full state.
        """
        n = candidate.shape[0]
        ci = np.asarray(self.constraint_indices)
        tol = 1e-12
        if ci.size == 0:
            return sp.eye(n, format='csr')

        # Build rhok per-index
        if rhok is None:
            rhok_full = np.ones(n, dtype=float)
        elif np.isscalar(rhok):
            rhok_full = np.full((n,), float(rhok), dtype=float)
        else:
            rhok_full = _to_numpy_if_needed(rhok)

        # Evaluate conf and its (optional) Jacobian at the provided current_state
        conf = _to_numpy_if_needed(self._call_con_force(current_state, t=t, Fk_val=Fk_val))
        if self.conf_jacobian_mode == 'full':
            J_conf = self._compute_jacobian(current_state, t=t, Fk_val=Fk_val)
        else:
            J_conf = None  # skip computing expensive Jacobian in 'none' mode

        # Helper 2x2 projector matrices
        P_ray_pp = 0.5 * np.array([[1.0,  1.0],
                                   [1.0,  1.0]])  # onto (1,1)
        P_ray_mp = 0.5 * np.array([[1.0, -1.0],
                                   [-1.0, 1.0]])  # onto (-1,1)
        P_zero   = np.zeros((2, 2))
        P_I      = np.eye(2)
        P_tie    = 0.5 * np.eye(2)

        # Optional numba-aided region classification (vectorized by pairs)
        use_nb = False
        if isinstance(getattr(self, 'use_numba', 'auto'), str):
            use_nb = (_NUMBA_OK and self.use_numba == 'auto')
        else:
            use_nb = bool(self.use_numba) and _NUMBA_OK

        # Precompute per-index v and z_tilde for classification (no augmented z)
        valid_mask = ci < n
        if not np.any(valid_mask):
            return sp.eye(n, format='csr')
        valid_indices = ci[valid_mask]
        v_arr = _to_numpy_if_needed(candidate[valid_indices]).astype(float, copy=False)
        zt_arr = (
            np.abs(_to_numpy_if_needed(current_state[valid_indices]))
            - _to_numpy_if_needed(rhok_full[valid_indices]) * _to_numpy_if_needed(conf[valid_indices])
        ).astype(float, copy=False)

        # Region codes: try numba classifier first when requested
        codes = None
        if use_nb and _NUMBA_OK and len(valid_indices) > 0:
            try:
                codes = classify_regions_nb(v_arr, zt_arr, tol)
            except Exception:
                codes = None

        if codes is None:
            # Pure NumPy vectorized classification
            scale = 1.0 + np.maximum(np.abs(v_arr), np.abs(zt_arr))
            tol_s = tol * scale
            mask_tip = (np.abs(v_arr) <= tol_s) & (np.abs(zt_arr) <= tol_s)
            mask_ray_pp = (~mask_tip) & (np.abs(zt_arr) < (v_arr - tol_s))
            mask_zero = (~mask_tip) & (~mask_ray_pp) & (np.abs(v_arr) < (-zt_arr - tol_s))
            mask_ray_mp = (~mask_tip) & (~mask_ray_pp) & (~mask_zero) & (np.abs(zt_arr) < (-v_arr - tol_s))
            mask_tie = (~mask_tip) & (~mask_ray_pp) & (~mask_zero) & (~mask_ray_mp) & (
                (np.abs(np.abs(zt_arr) - v_arr) <= tol_s) | (np.abs(np.abs(v_arr) + zt_arr) <= tol_s)
            )
            codes = np.full(len(valid_indices), 5, dtype=int)  # 5 => identity
            codes[mask_tip] = 0
            codes[mask_ray_pp] = 2
            codes[mask_zero] = 0
            codes[mask_ray_mp] = 3
            codes[mask_tie] = 4

        # Prepare dense rows for modified indices (store only nonzeros later)
        modified_rows = {}
        for k, idx in enumerate(valid_indices):
            code = int(codes[k])
            if code == 0:
                P = P_zero
            elif code == 1:
                P = P_I
            elif code == 2:
                P = P_ray_pp
            elif code == 3:
                P = P_ray_mp
            elif code == 4:
                P = P_tie
            else:
                P = P_I
            if self.conf_jacobian_mode == 'full':
                row_gi = _to_numpy_if_needed(J_conf[idx])
            else:
                row_gi = 0.0
            # dv/dy
            col_v = np.zeros((n,), dtype=float); col_v[idx] = 1.0
            # d z_tilde / dy = sign(current_state[idx]) e_idx - rhok[idx]*J_conf[idx,:]
            col_zt = np.zeros((n,), dtype=float)
            sign_val = np.sign(current_state[idx])
            if sign_val == 0:
                sign_val = 1.0  # convention at the kink
            col_zt[idx] = sign_val
            if self.conf_jacobian_mode == 'full':
                col_zt -= rhok_full[idx] * row_gi
            # Only the v-row (top) contributes to the non-augmented state
            row_top = P[0, 0] * col_v + P[0, 1] * col_zt
            modified_rows[idx] = row_top

        # Assemble CSR directly
        data = []
        indices_list = []
        indptr = [0]
        nz_tol = 1e-15
        for r in range(n):
            row = modified_rows.get(r)
            if row is None:
                data.append(1.0)
                indices_list.append(r)
            else:
                nz = np.flatnonzero(np.abs(row) > nz_tol)
                if nz.size:
                    data.extend(row[nz])
                    indices_list.extend(nz.tolist())
            indptr.append(len(data))
        data = np.array(data, dtype=float)
        indices_arr = np.array(indices_list, dtype=int)
        D_csr = sp.csr_matrix((data, indices_arr, np.array(indptr)), shape=(n, n))
        return D_csr


##############################################################################
# MuScaledSOCProjection — μ-scaled second-order cone projector
##############################################################################
class MuScaledSOCProjection(Projection):
    """
    Project one or more SOC blocks (s, w) onto the μ-scaled second-order cone
        K_μ = { (s, w) : s >= 0, ||w|| <= μ s }.

    μ is supplied via a callable at construction, mirroring `con_force_func`
    in CoulombProjection. It may return a scalar (broadcast to all blocks) or
    an array of length equal to the number of blocks.

    Blocks can be specified explicitly as a list of tuples (s_index, w_indices)
    or implicitly via `component_slices`, where each slice denotes a contiguous
    block [s, w...] with s at the first index in the slice. Alternatively, pass
    a dynamic ``blocks_func`` that will be called each iteration with
    ``(t, candidate, current_state, prev_state, Fk_val)`` and must return a list
    of blocks (tuples or slices) for that call.

    You may also pass a ``prefill_func`` which, given the same call signature
    plus the resolved blocks, can modify a working copy of the candidate before
    the projection (e.g., set w = r(y) − Φ(y) y explicitly). Its signature may
    optionally accept the current step size ``h`` and ``rhok`` keyword:
    ``prefill_func(t, candidate, current_state, prev_state, blocks, Fk_val, step_size=None, rhok=None)``.

    Returns
    -------
    project(...): ndarray
        A full-sized projected state vector. For each block, both the s and w
        entries are overwritten by the projector result; non-block entries are
        left unchanged (except for any edits your ``prefill_func`` performs).

    Notes
    -----
    - ``tangent_cone`` assembles a Clarke selection of the projector itself.
      If you use a nontrivial ``prefill_func`` that depends on state, its
      derivative is not composed into the returned Jacobian (chain rule is
      omitted) — this mirrors the lightweight pattern used by other simple
      projectors. For fully consistent Jacobians, provide your mapping inside
      the model residual or extend this class to inject the chain terms.
    - ``rhok`` is accepted for interface compatibility but not used.
    """

    def __init__(self, *, blocks=None, component_slices=None,
                 blocks_func=None, prefill_func=None, constraint_indices=None, get_mu =None):
        super().__init__(component_slices=component_slices)
        # self.blocks = self._normalize_blocks(blocks, component_slices) if (blocks is not None or component_slices) else None
        self.blocks_func = blocks_func
        self.prefill_func = prefill_func
        self.constraint_indices = constraint_indices
        self.get_mu = get_mu
    # ---------- helpers ----------
    @staticmethod
    def _normalize_blocks(blocks, component_slices):
        norm = []
        if blocks is not None:
            for blk in blocks:
                if isinstance(blk, slice):
                    idx = np.arange(blk.start, blk.stop)
                    if idx.size < 2:
                        raise ValueError("SOC block slice must be length >= 2 (s + at least 1 w)")
                    s_idx = int(idx[0])
                    w_idx = idx[1:]
                    norm.append((s_idx, np.asarray(w_idx, dtype=int)))
                else:
                    s_idx, w_idx = blk
                    # allow s_idx to be None to indicate a virtual s (provided by s_value_func)
                    w_idx = np.asarray(w_idx, dtype=int)
                    if w_idx.size < 1:
                        raise ValueError("SOC block must include at least one tangential component")
                    s_idx_norm = int(s_idx) if s_idx is not None else None
                    norm.append((s_idx_norm, w_idx))
        elif component_slices:
            for sl in component_slices:
                if not hasattr(sl, 'start'):
                    raise ValueError("component_slices must be slice objects for SOC blocks")
                idx = np.arange(sl.start, sl.stop)
                if idx.size < 2:
                    raise ValueError("SOC block slice must be length >= 2 (s + at least 1 w)")
                s_idx = int(idx[0])
                w_idx = idx[1:]
                norm.append((s_idx, np.asarray(w_idx, dtype=int)))
        else:
            raise ValueError("Provide either blocks or component_slices to define SOC blocks.")
        return norm

    def _eval_mu_per_block(self, y, t=None, Fk_val=None, nb_blocks=None):
        # Flexible μ call: (y,t,Fk_val) | (y,t) | (y)
        try:
            mu_val = self.mu_func(y, t, Fk_val)
        except TypeError:
            try:
                mu_val = self.mu_func(y, t)
            except TypeError:
                mu_val = self.mu_func(y)
        mu_arr = np.atleast_1d(np.asarray(mu_val, float))
        # Decide target number of blocks
        if nb_blocks is None:
            nb_blocks = len(self.blocks) if self.blocks is not None else None
        if mu_arr.size == 1:
            # caller will broadcast if nb_blocks known
            return mu_arr.astype(float)
        if nb_blocks is not None and mu_arr.size == nb_blocks:
            return mu_arr.astype(float)
        if self.blocks is not None and mu_arr.size == len(self.blocks):
            return mu_arr.astype(float)
        raise ValueError("mu_func must return a scalar or an array sized to the number of SOC blocks for this call")

    @staticmethod
    def _proj_mu_scaled_soc(z, mu, return_jacobian=False, eps=1e-30):
        z = np.asarray(z, float)
        s = float(z[0])
        w = np.asarray(z[1:], float)
        m = w.size
        r = float(np.linalg.norm(w))

        # Inside K_μ -> identity
        if (s >= 0.0) and (r <= mu * s):
            return z.copy()

        # Inside polar K_μ° = { (t,v): t <= - μ ||v|| }
        if (s <= 0.0) and (r <= (-s) / mu if mu > 0 else 0.0):
            return np.zeros_like(z)

        # Boundary closed form
        alpha = 1.0 / (1.0 + mu * mu)
        r_eff = max(r, eps)
        what  = w / r_eff
        beta  = s + mu * r

        p_s = alpha * beta
        p_w = p_s * (mu * what)

        return np.hstack([p_s, p_w])


    # ---------- Projection API ----------
    def project(self, current_state, z, rhok=None, t=None, Fk_val=None, prev_state=None, step_size=None, **kw):
        y = np.asarray(current_state, float)
        z_work = np.asarray(z, float).copy()

        g = y[3]  # q_y is the vertical gap
        print(f'gap: {g}')
        if g > 0:  # No contact
            # print('no contact')
            # z_work[self.constraint_indices] = 0.0
            return z_work

        if self.prefill_func is not None:
            try:
                filled = self.prefill_func(t, z_work, y, prev_state, Fk_val, step_size, rhok=rhok)
            except TypeError:
                try:
                    filled = self.prefill_func(t, z_work, y, prev_state, Fk_val, step_size)
                except TypeError:
                    try:
                        filled = self.prefill_func(t, z_work, y, prev_state, Fk_val, rhok=rhok)
                    except TypeError:
                        filled = self.prefill_func(t, z_work, y, prev_state, Fk_val)
            z_work[self.constraint_indices] = np.asarray(filled, float)
        
        mu = self.get_mu(y, t=t, Fk_val=Fk_val)
        
        block_TN = z_work[self.constraint_indices].astype(float)   # [T,N]
        z_soc = np.array([block_TN[1], block_TN[0]], float)        # [N,T]

        # project in SOC space (correct μ≠1 polar branch inside)
        # z_soc_proj = self._proj_mu_scaled_soc(z_soc, 1/mu)           # ũ^{k+1} in [N,T]
        # project onto the DUAL cone because we are updating \hat u
        if mu == 0.0:
            # K_{1/μ} with μ=0 ⇒ K_∞ : clamp normal only
            z_soc_proj = np.array([max(z_soc[0], 0.0), z_soc[1]], float)
        else:
            z_soc_proj = self._proj_mu_scaled_soc(z_soc, 1.0/mu)  # <-- NOTE 1/mu

            # z_soc_proj = self._proj_mu_scaled_soc(z_soc, 1.0/mu)  # <-- NOTE 1/mu
        # invert Θ once to get u
        e = 0.5
        theta = 0.5
        uN_prev = prev_state[1] if prev_state is not None else 0.0
        tilde_u_N, tilde_u_T = z_soc_proj[0], z_soc_proj[1]
        u_T = tilde_u_T
        # gamma= 1/2*(1+e)-1

        u_N = tilde_u_N - e * uN_prev - mu * abs(tilde_u_T)

        z_work[self.constraint_indices] = np.array([u_T, u_N], float)  # back to [T,N]
        return z_work

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None, prev_stat=None, step_size=None, **kw):
        y = np.asarray(current_state, float)
        z_work = np.asarray(candidate, float)
        n = z_work.size

        # Resolve blocks (dynamic allowed)
        if self.blocks_func is not None:
            blocks_raw = self.blocks_func(t, z_work, y, prev_state, Fk_val)
            blocks = self._normalize_blocks(blocks_raw, None)
        else:
            if self.blocks is None:
                return sp.eye(n, format='csr')
            blocks = self.blocks

        # Optional prefill for evaluating local Jacobians at the right z
        if self.prefill_func is not None:
            try:
                z_work = np.asarray(
                    self.prefill_func(t, z_work.copy(), y, prev_state, blocks, Fk_val, step_size, rhok=rhok),
                    float,
                )
            except TypeError:
                try:
                    z_work = np.asarray(
                        self.prefill_func(t, z_work.copy(), y, prev_state, blocks, Fk_val, step_size),
                        float,
                    )
                except TypeError:
                    try:
                        z_work = np.asarray(
                            self.prefill_func(t, z_work.copy(), y, prev_state, blocks, Fk_val, rhok=rhok),
                            float,
                        )
                    except TypeError:
                        z_work = np.asarray(
                            self.prefill_func(t, z_work.copy(), y, prev_state, blocks, Fk_val),
                            float,
                        )

        mu_blocks = self._eval_mu_per_block(y, t=t, Fk_val=Fk_val, nb_blocks=len(blocks))
        if mu_blocks.size == 1:
            mu_blocks = np.full(len(blocks), float(mu_blocks[0]))

        # Assemble sparse CSR: start with identity rows, replace block rows
        data = []
        indices_list = []
        indptr = [0]

        # Map from row -> (block, local_row, idx)
        block_row_map = {}
        for b, (s_idx, w_idx) in enumerate(blocks):
            if s_idx is None:
                # Only w rows will be replaced; mark them as local rows 1..m
                idx = np.asarray(w_idx, dtype=int)
                # store with an offset marker: local row in full Jb corresponds to 1..m
                for offset, r in enumerate(idx, start=1):
                    block_row_map[int(r)] = (b, offset, np.r_[[-1], w_idx])  # -1 placeholder for s
            else:
                idx = np.r_[s_idx, w_idx]
                for local_r, r in enumerate(idx):
                    block_row_map[int(r)] = (b, local_r, idx)

        nz_tol = 1e-15
        for r in range(n):
            map_entry = block_row_map.get(r)
            if map_entry is None:
                data.append(1.0)
                indices_list.append(r)
                indptr.append(len(data))
                continue

            b, local_r, idx = map_entry
            mu_b = float(mu_blocks[b])
            # form z_block appropriately
            if idx[0] == -1:  # virtual s path
                # obtain s value (from s_value_func if provided, otherwise from first component slice or fallback)
                if self.s_value_func is not None:
                    try:
                        s_vals = self.s_value_func(t, z_work, y, prev_state, blocks, Fk_val, rhok=rhok)
                    except TypeError:
                        s_vals = self.s_value_func(t, z_work, y, prev_state, blocks, Fk_val)
                    s_val = float(np.atleast_1d(s_vals)[b])
                else:
                    if self.component_slices:
                        u_sl = self.component_slices[0]
                        s_val = float(np.linalg.norm(y[u_sl]))
                    else:
                        k = len(idx) - 1  # number of w components
                        s_val = float(np.linalg.norm(y[:k]))
                z_block = np.hstack([s_val, z_work[idx[1:]]])
                _, Jb = self._proj_mu_scaled_soc(z_block, mu_b, return_jacobian=True)
                row_vals = Jb[local_r, 1:]  # only w columns correspond to state columns idx[1:]
                nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
                if nz.size:
                    data.extend(row_vals[nz])
                    indices_list.extend(idx[1:][nz].tolist())
                indptr.append(len(data))
                continue
            z_block = z_work[idx]
            _, Jb = self._proj_mu_scaled_soc(z_block, mu_b, return_jacobian=True)
            row_vals = Jb[local_r, :]
            nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
            if nz.size:
                data.extend(row_vals[nz])
                indices_list.extend(idx[nz].tolist())
            indptr.append(len(data))

        return sp.csr_matrix((np.array(data, float), np.array(indices_list, int), np.array(indptr)), shape=(n, n))

##############################################################################
# GeneralMoreauVIProjection  —  drop-in, solver-compatible
##############################################################################
class GeneralMoreauVIProjection(Projection):
    """
    Domain-agnostic unilateral projector for VI/SSN loops.

    Callbacks:
      gap(t,y)   -> (m,)    signed gaps (active if <= gap_tol)
      u_map(t,y) -> (m,)    channel outputs (e.g., normal velocity)
      J_u(t,y)   -> (m,n)   du/dy at y
      G_apply(t,y,lam_full)->(n,)  Δy from channel impulse vector λ ∈ R^m

    We solve:  find λ ≥ 0 ⟂ (W λ + b) ≥ 0, with  W = J_u G,  b = u_free + E u_prev
    and return  y_new = y_bar + G λ,  where y_bar = candidate = y_k − ρ R(y_k).

    NOTE: accepts rhok= (what the solver passes) and also rho= (legacy).
    """

    def __init__(self, gap, u_map, J_u, G_apply,
                 e=0.0, gap_tol=0.0,
                 lcp_maxit=80, lcp_tol=1e-12,
                 tc_tol=1e-12,
                 component_slices=None):
        super().__init__(component_slices=component_slices)
        self.gap = gap
        self.u_map = u_map
        self.J_u = J_u
        self.G_apply = G_apply
        self.e = e
        self.gap_tol = float(gap_tol)
        self.lcp_maxit = int(lcp_maxit)
        self.lcp_tol = float(lcp_tol)
        self.tc_tol = float(tc_tol)

    # ---------- LCP helper (PG on R_+) ----------
    @staticmethod
    def _solve_Rplus_LCP_pg(W, b, maxit, tol):
        r = np.zeros_like(b)
        alpha = 1.0 / np.clip(np.diag(W), 1e-16, np.inf)
        for _ in range(maxit):
            z = W @ r + b
            r_new = np.maximum(0.0, r - alpha * z)
            if np.linalg.norm(r_new - r) < tol:
                return r_new
            r = r_new
        return r

    # ---------- shared assembly for W and b on the active set ----------
    def _build_W_b(self, t, y_bar, prev_state, active_idx_full):
        J = np.asarray(self.J_u(t, y_bar), float)  # (m, n)
        m = J.shape[0]
        act = np.asarray(active_idx_full, int)
        ma = act.size

        def restrict(v):               # R^m -> R^{ma}
            v = np.asarray(v, float)
            return v[act]

        def embed(v_a):                # R^{ma} -> R^m
            w = np.zeros(m)
            w[act] = np.asarray(v_a, float)
            return w

        u_free_full = np.asarray(self.u_map(t, y_bar), float)
        u_free = restrict(u_free_full)

        if prev_state is not None and np.any(self.e):
            e = np.asarray(self.e, float)
            if e.size == 1:
                e = np.full(m, float(e))
            u_prev = restrict(self.u_map(t, prev_state))
            b = u_free + e[act] * u_prev
        else:
            b = u_free.copy()

        # W = J_u G (m_a × m_a) by columns
        W = np.zeros((ma, ma))
        for j in range(ma):
            ej_full = embed(np.eye(ma)[:, j])
            dy = self.G_apply(t, y_bar, ej_full)   # (n,)
            du_full = J @ dy                       # (m,)
            W[:, j] = restrict(du_full)            # (m_a,)
        return W, b, J, restrict, embed, act

    # ---------- projection ----------
    def project(self, state, state_minus_rhoR,
                rhok=None, t=None, Fk_val=None, prev_state=None, **kw):
        """
        Accepts rhok= (preferred) or rho= (ignored value; for legacy callers).
        """
        # Legacy compatibility: accept rho= if someone passes it
        if rhok is None and 'rho' in kw:
            rhok = kw['rho']  # value unused; kept for interface symmetry

        y_it  = np.asarray(state, float)
        y_bar = np.asarray(state_minus_rhoR, float).copy()

        # Activate if either iterate or candidate is inside
        g_bar = np.asarray(self.gap(t, y_bar), float)
        g_it  = np.asarray(self.gap(t, y_it),  float)
        active = np.where(np.minimum(g_bar, g_it) <= self.gap_tol)[0]
        if active.size == 0:
            return y_bar

        W, b, J, restrict, embed, act = self._build_W_b(t, y_bar, prev_state, active)
        lam_a = self._solve_Rplus_LCP_pg(W, b, self.lcp_maxit, self.lcp_tol)
        lam_full = embed(lam_a)
        dy = self.G_apply(t, y_bar, lam_full)
        return y_bar + dy

    # ---------- tangent cone (Clarke selection) ----------
    def tangent_cone(self, candidate, current_state,
                     rhok=None, t=None, Fk_val=None, prev_state=None, **kw):
        # accept rho= via **kw as well
        if rhok is None and 'rho' in kw:
            rhok = kw['rho']

        y_bar = np.asarray(candidate, float)
        y_it  = np.asarray(current_state, float)
        n = y_bar.size

        g_bar = np.asarray(self.gap(t, y_bar), float)
        g_it  = np.asarray(self.gap(t, y_it),  float)
        active = np.where(np.minimum(g_bar, g_it) <= self.gap_tol)[0]
        if active.size == 0:
            return sp.eye(n, format='csr')

        prev_for_b = prev_state if prev_state is not None else y_it
        W, b, J, restrict, embed, act = self._build_W_b(t, y_bar, prev_for_b, active)
        lam_a = self._solve_Rplus_LCP_pg(W, b, self.lcp_maxit, self.lcp_tol)
        u_free = restrict(self.u_map(t, y_bar))
        u_plus = u_free + W @ lam_a

        scale = 1.0 + np.maximum(np.abs(u_plus), np.abs(lam_a))
        tol = self.tc_tol * scale

        A_mask   = (u_plus <= tol)
        tie_mask = (np.abs(u_plus) <= tol) & (lam_a <= tol)

        if not np.any(A_mask):
            return sp.eye(n, format='csr')

        A_idx_in_active = np.where(A_mask)[0]
        A_full = act[A_idx_in_active]
        W_AA = W[np.ix_(A_idx_in_active, A_idx_in_active)]

        try:
            W_AA_inv = np.linalg.inv(W_AA)
        except np.linalg.LinAlgError:
            print("we are singular!")
            eps = 1e-12 * max(1.0, np.linalg.norm(W_AA, ord=2))
            W_AA_inv = np.linalg.inv(W_AA + eps * np.eye(W_AA.shape[0]))

        J_full = np.asarray(self.J_u(t, y_bar), float)
        J_A = J_full[A_full, :]

        G_A = np.zeros((n, A_idx_in_active.size))
        for j, j_in_act in enumerate(A_idx_in_active):
            ej_full = np.zeros(J_full.shape[0])
            ej_full[act[j_in_act]] = 1.0
            G_A[:, j] = self.G_apply(t, y_bar, ej_full)

        alpha = np.ones(A_idx_in_active.size)
        alpha[tie_mask[A_idx_in_active]] = 0.5
        WA_inv_row_scaled = (alpha[:, None]) * W_AA_inv

        D_dense = np.eye(n) + G_A @ ( - WA_inv_row_scaled @ J_A )
        return sp.csr_matrix(D_dense)
