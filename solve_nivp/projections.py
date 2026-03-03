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

import inspect

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

    def __init__(self, component_slices=None):
        super().__init__(component_slices=component_slices)
        self._eye_cache = {}

    def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
        return candidate

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        n = candidate.shape[0]
        eye = self._eye_cache.get(n)
        if eye is None:
            eye = sp.eye(n, format='csr')
            self._eye_cache[n] = eye
        return eye

    def project_batch(self, current_state, candidates, rhok=None, t=None, Fk_val=None):
        return np.asarray(candidates)


##############################################################################
# _ConstraintBlock  (internal helper for AlgebraicConstraintProjection)
##############################################################################
def _count_required_args(fn):
    """Count the number of **required** positional parameters of *fn*.

    Parameters that have default values (common closure-capture pattern,
    e.g. ``lambda p, _C=matrix: _C @ p``) are *not* counted.
    Returns ``None`` if introspection fails (e.g. C built-ins).
    """
    if fn is None:
        return 0
    try:
        sig = inspect.signature(fn)
        return sum(
            1 for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    except (ValueError, TypeError):
        return None          # fall back to runtime probing


class _ConstraintBlock:
    """Per-constraint callable with arity-dispatch cache.

    Each block stores one algebraic relation ``q = g(y)`` together with
    its analytical Jacobian (or finite-difference fallback) and the
    index slices that locate ``y`` and ``q`` inside the full state
    vector.

    Arity is detected at construction time via ``inspect.signature`` so
    that user lambdas with closure-captured defaults (e.g.
    ``lambda p, _C=matrix: _C @ p``) are correctly identified as
    single-argument functions.
    """
    __slots__ = ('g', 'dg_dy', 'y_slice', 'q_slice', 'fd_eps',
                 '_g_nargs', '_dg_nargs')

    def __init__(self, g, dg_dy, y_slice, q_slice, fd_eps=1e-8):
        self.g = g
        self.dg_dy = dg_dy
        self.y_slice = y_slice
        self.q_slice = q_slice
        self.fd_eps = float(fd_eps)
        self._g_nargs = _count_required_args(g)
        self._dg_nargs = _count_required_args(dg_dy)

    # ---- flexible calling helpers ----
    def call_g(self, y_sub, t=None, Fk_val=None):
        """Evaluate ``g(y_sub)`` with auto-detected arity."""
        if self.g is None:
            s = self.q_slice
            start = s.start or 0
            stop = s.stop
            step = s.step or 1
            n_q = max(0, (stop - start + step - 1) // step) if stop is not None else y_sub.size
            return np.zeros(n_q, dtype=float)
        n = self._g_nargs
        if n is not None:
            if n >= 3:
                return np.asarray(self.g(y_sub, t, Fk_val), dtype=float)
            elif n == 2:
                return np.asarray(self.g(y_sub, t), dtype=float)
            else:
                return np.asarray(self.g(y_sub), dtype=float)
        # Fallback: runtime probing (only when inspect.signature failed)
        try:
            r = self.g(y_sub, t, Fk_val); self._g_nargs = 3
            return np.asarray(r, dtype=float)
        except TypeError:
            try:
                r = self.g(y_sub, t); self._g_nargs = 2
                return np.asarray(r, dtype=float)
            except (TypeError, ValueError):
                r = self.g(y_sub); self._g_nargs = 1
                return np.asarray(r, dtype=float)

    def call_dg(self, y_sub, t=None, Fk_val=None):
        """Evaluate ``dg/dy`` or fall back to finite differences."""
        if self.dg_dy is None:
            return self._fd_jacobian(y_sub, t=t, Fk_val=Fk_val)
        n = self._dg_nargs
        if n is not None:
            if n >= 3:
                return np.asarray(self.dg_dy(y_sub, t, Fk_val), dtype=float)
            elif n == 2:
                return np.asarray(self.dg_dy(y_sub, t), dtype=float)
            else:
                return np.asarray(self.dg_dy(y_sub), dtype=float)
        # Fallback: runtime probing (only when inspect.signature failed)
        try:
            r = self.dg_dy(y_sub, t, Fk_val); self._dg_nargs = 3
            return np.asarray(r, dtype=float)
        except TypeError:
            try:
                r = self.dg_dy(y_sub, t); self._dg_nargs = 2
                return np.asarray(r, dtype=float)
            except (TypeError, ValueError):
                r = self.dg_dy(y_sub); self._dg_nargs = 1
                return np.asarray(r, dtype=float)

    def _fd_jacobian(self, y_sub, t=None, Fk_val=None):
        """Forward-difference Jacobian of g w.r.t. y."""
        y0 = np.asarray(y_sub, dtype=float)
        g0 = self.call_g(y0, t=t, Fk_val=Fk_val)
        n_y, n_q = y0.size, g0.size
        J = np.empty((n_q, n_y), dtype=float)
        eps = self.fd_eps
        for j in range(n_y):
            y_p = y0.copy()
            y_p[j] += eps
            J[:, j] = (self.call_g(y_p, t=t, Fk_val=Fk_val) - g0) / eps
        return J


##############################################################################
# AlgebraicConstraintProjection
##############################################################################
class AlgebraicConstraintProjection(Projection):
    r"""Project onto one or more algebraic constraint manifolds.

    **Single-constraint form** (original API):

    Given :math:`z = [y;\, q]` and the relation :math:`q = g(y)`, the
    projection overwrites the algebraic components with their exact
    manifold values.

    **Multi-constraint form**:

    Given :math:`z = [\ldots, q_1, \ldots, q_2, \ldots]` and multiple
    independent relations :math:`q_k = g_k(y_k)` (each algebraic block
    may depend on a *different* subset of the state), all constraints
    are enforced simultaneously.

    Parameters
    ----------
    g : callable, optional
        Single constraint map ``g(y) -> q``.  Mutually exclusive with
        ``constraints``.
    dg_dy : callable or None, optional
        Analytical Jacobian for the single-constraint form.
    y_slice : slice, optional
        Differential-DOF index range (single-constraint form).
    q_slice : slice, optional
        Algebraic-DOF index range (single-constraint form).
    constraints : list[dict], optional
        List of constraint specifications.  Each dict must contain:

        * ``'g'`` — constraint map (same calling conventions as ``g``),
        * ``'y_slice'`` — slice locating the inputs in the full state,
        * ``'q_slice'`` — slice locating the outputs in the full state.

        Optional keys:

        * ``'dg_dy'`` — analytical Jacobian (defaults to FD),
        * ``'fd_eps'`` — FD step size for this block (defaults to the
          global ``fd_eps``).

        The ``q_slice`` ranges **must not overlap** across constraints.
        The ``y_slice`` ranges *may* overlap (e.g. two constraints that
        both depend on the same differential DOFs).

        Constraints are applied in list order; if a later constraint's
        ``y_slice`` overlaps an earlier constraint's ``q_slice``, the
        earlier constraint is applied first (sequential projection).
    component_slices : list[slice], optional
        Block partition forwarded to the base class.
    fd_eps : float, default 1e-8
        Default finite-difference step size.

    Notes
    -----
    * Works with both the **VI** and **semismooth Newton** solvers.
    * The ``tangent_cone`` returns a sparse CSR matrix whose structure
      is cached after the first call for efficiency.
    * Designed for singular mass matrices with zero rows on the
      algebraic DOFs.

    Examples
    --------
    Single constraint (backward-compatible):

    >>> import numpy as np
    >>> from solve_nivp import AlgebraicConstraintProjection
    >>> C = np.array([[2.0, 0.0], [0.0, 3.0]])
    >>> proj = AlgebraicConstraintProjection(
    ...     g=lambda y: C @ y,
    ...     dg_dy=lambda y: C,
    ...     y_slice=slice(0, 2),
    ...     q_slice=slice(2, 4),
    ... )
    >>> proj.project(np.zeros(4), np.array([1., 1., 5., 5.]))
    array([1., 1., 2., 3.])

    Multiple constraints (e.g. poromechanics DAE):

    >>> C_qp  = np.array([[1.0, 0.5]])        # λ_q  = C_qp  @ p
    >>> C_su  = np.array([[0.3], [0.7]])       # λ_σ  = C_su  @ u
    >>> proj = AlgebraicConstraintProjection(constraints=[
    ...     dict(g=lambda p: C_qp @ p, dg_dy=lambda p: C_qp,
    ...          y_slice=slice(0, 2), q_slice=slice(3, 4)),
    ...     dict(g=lambda u: C_su @ u, dg_dy=lambda u: C_su,
    ...          y_slice=slice(2, 3), q_slice=slice(4, 6)),
    ... ])
    """

    def __init__(self, g=None, dg_dy=None, y_slice=None, q_slice=None,
                 constraints=None, component_slices=None, fd_eps=1e-8):
        super().__init__(component_slices=component_slices)
        fd_eps = float(fd_eps)

        # --- Normalise both API forms into self._blocks ---
        if constraints is not None:
            if g is not None or y_slice is not None or q_slice is not None:
                raise ValueError(
                    "Specify either (g, y_slice, q_slice) or constraints=[], "
                    "not both.")
            if not constraints:
                raise ValueError("constraints list must not be empty.")
            self._blocks = []
            for i, spec in enumerate(constraints):
                if 'y_slice' not in spec or 'q_slice' not in spec:
                    raise ValueError(
                        f"Constraint {i}: both 'y_slice' and 'q_slice' "
                        f"are required.")
                self._blocks.append(_ConstraintBlock(
                    g=spec.get('g'),
                    dg_dy=spec.get('dg_dy'),
                    y_slice=spec['y_slice'],
                    q_slice=spec['q_slice'],
                    fd_eps=spec.get('fd_eps', fd_eps),
                ))
            # Validate non-overlapping q_slices
            self._validate_q_slices()
        else:
            if y_slice is None or q_slice is None:
                raise ValueError(
                    "Both y_slice and q_slice must be provided "
                    "(or use constraints=[...]).")
            self._blocks = [_ConstraintBlock(
                g=g, dg_dy=dg_dy,
                y_slice=y_slice, q_slice=q_slice, fd_eps=fd_eps,
            )]

        # Backward-compat attributes (single-constraint only)
        if len(self._blocks) == 1:
            blk = self._blocks[0]
            self.g = blk.g
            self.dg_dy = blk.dg_dy
            self.y_slice = blk.y_slice
            self.q_slice = blk.q_slice
        else:
            self.g = self.dg_dy = self.y_slice = self.q_slice = None

        self.fd_eps = fd_eps

        # The algebraic projection enforces q = g(y) regardless of rhok,
        # so Lipschitz-based lam/rho adaptation in the nonlinear solver
        # is superfluous.  This flag lets the solver skip it automatically.
        self.rho_independent = True

        # Tangent-cone structure cache (built lazily on first call)
        self._tc_cached = None        # CSR matrix
        self._tc_n = None             # state dimension
        self._tc_jg_positions = None  # list of int-arrays into CSR .data

        # Constraint-patch cache for direct Newton solver (built lazily)
        self._patch_cached = None     # CSR matrix
        self._patch_n = None          # state dimension
        self._patch_dg_positions = None  # list of int-arrays into CSR .data

    # ------------------------------------------------------------------
    def _validate_q_slices(self):
        """Ensure no two constraints write to the same DOF index."""
        seen = set()
        for k, blk in enumerate(self._blocks):
            # Resolve slice to concrete indices (use a generous upper bound)
            ub = max((blk.q_slice.stop or 0), (blk.y_slice.stop or 0)) + 1
            q_idx = set(range(*blk.q_slice.indices(ub)))
            overlap = seen & q_idx
            if overlap:
                raise ValueError(
                    f"Constraint {k}: q_slice indices {sorted(overlap)} "
                    f"overlap with a previous constraint.")
            seen |= q_idx

    # ---- Projection API ----
    def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
        out = candidate.copy() if isinstance(candidate, np.ndarray) else np.array(candidate, dtype=float)
        for blk in self._blocks:
            y_sub = out[blk.y_slice]
            out[blk.q_slice] = blk.call_g(y_sub, t=t, Fk_val=Fk_val)
        return out

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        r"""Clarke generalized derivative of the projection.

        Non-constrained rows pass through (identity).  For each
        constraint block *k*, algebraic row :math:`q_i` has entries
        :math:`(\partial g_k / \partial y_k)_{i,j}` on the
        ``y_slice`` columns and zero elsewhere.

        Returns
        -------
        D : scipy.sparse.csr_matrix, shape (n, n)
        """
        z = np.asarray(candidate, dtype=float)
        n = z.size

        # Compute all Jg blocks
        Jg_list = []
        for blk in self._blocks:
            Jg = blk.call_dg(z[blk.y_slice], t=t, Fk_val=Fk_val)
            if sp.issparse(Jg):
                Jg = Jg.toarray()
            Jg_list.append(np.atleast_2d(Jg))

        if self._tc_cached is None or self._tc_n != n:
            self._build_tangent_cache(n, Jg_list)
        else:
            # Hot path: update each block's values in-place
            for pos, Jg in zip(self._tc_jg_positions, Jg_list):
                if pos.size > 0:
                    self._tc_cached.data[pos] = Jg.ravel()

        return self._tc_cached

    def _build_tangent_cache(self, n, Jg_list):
        """One-time construction of the tangent-cone CSR and index maps.

        The sparsity structure is constant (dense Jg blocks for every
        constraint), so subsequent ``tangent_cone`` calls only overwrite
        the value arrays — no COO rebuild or ``tocsr()`` conversion.
        """
        # Collect all constrained indices across blocks
        all_q_set = set()
        for blk in self._blocks:
            all_q_set.update(range(*blk.q_slice.indices(n)))

        # Identity rows: everything NOT constrained
        identity_rows = np.array(
            sorted(set(range(n)) - all_q_set), dtype=int)

        rows_list, cols_list, vals_list = [], [], []

        # Identity entries
        if identity_rows.size > 0:
            rows_list.append(identity_rows)
            cols_list.append(identity_rows)
            vals_list.append(np.ones(identity_rows.size))

        # Per-block Jg entries (full dense block for constant nnz)
        for blk, Jg in zip(self._blocks, Jg_list):
            q_idx = np.arange(*blk.q_slice.indices(n))
            y_idx = np.arange(*blk.y_slice.indices(n))
            n_qk, n_yk = q_idx.size, y_idx.size
            if n_qk > 0 and n_yk > 0:
                rows_list.append(np.repeat(q_idx, n_yk))
                cols_list.append(np.tile(y_idx, n_qk))
                vals_list.append(Jg.ravel().copy())

        all_r = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
        all_c = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)
        all_v = np.concatenate(vals_list) if vals_list else np.array([], dtype=float)

        D = sp.coo_matrix((all_v, (all_r, all_c)), shape=(n, n)).tocsr()
        D.sort_indices()

        # Map each block's Jg entries → positions in D.data
        # Vectorised: one np.searchsorted call per row (not per element).
        self._tc_jg_positions = []
        for blk in self._blocks:
            q_idx = np.arange(*blk.q_slice.indices(n))
            y_idx = np.arange(*blk.y_slice.indices(n))
            n_qk, n_yk = q_idx.size, y_idx.size
            if n_qk > 0 and n_yk > 0:
                jg_pos = np.empty(n_qk * n_yk, dtype=np.intp)
                for k, qi in enumerate(q_idx):
                    rs = D.indptr[qi]
                    row_cols = D.indices[rs:D.indptr[qi + 1]]
                    jg_pos[k * n_yk:(k + 1) * n_yk] = rs + np.searchsorted(row_cols, y_idx)
                self._tc_jg_positions.append(jg_pos)
            else:
                self._tc_jg_positions.append(np.array([], dtype=np.intp))

        self._tc_cached = D
        self._tc_n = n

    # ------------------------------------------------------------------
    # Fast-path helpers for the direct Newton solver
    # ------------------------------------------------------------------

    @property
    def constraint_q_slices(self):
        """Return a list of ``q_slice`` objects — one per constraint block."""
        return [blk.q_slice for blk in self._blocks]

    def build_constraint_patch(self, y, n, t=None, Fk_val=None, **kwargs):
        r"""Build a sparse matrix that fills the zero constraint rows of
        an iteration-matrix Jacobian with the algebraic constraint equations.

        For each constraint block *k* with :math:`q_k = g_k(y_{s_k})`:

        * row :math:`q_i`, col :math:`q_i` → +1  (identity on constraint diagonal)
        * row :math:`q_i`, col :math:`y_j` → :math:`-\partial g_{k,i}/\partial y_j`

        The returned matrix has **zero field rows** — it is intended to
        be *added* to an iteration matrix that already has zero
        constraint rows (because the mass / stiffness rows were stripped).

        Parameters
        ----------
        y : ndarray, shape (n,)
            Current state (needed for state-dependent dg/dy).
        n : int
            Full system dimension.
        t, Fk_val : optional
            Forwarded to ``dg_dy``.
        **kwargs : dict
            Extra context forwarded by the solver.  Recognised keys
            include ``step_size`` and ``prev_state`` (used by
            rate-dependent constraint subclasses such as
            Kelvin--Voigt projections).

        Returns
        -------
        patch : scipy.sparse.csr_matrix, shape (n, n)
        """
        if self._patch_cached is not None and self._patch_n == n:
            # Hot path: only update the dg/dy values in-place
            for blk, pos in zip(self._blocks, self._patch_dg_positions):
                Jg = blk.call_dg(y[blk.y_slice], t=t, Fk_val=Fk_val)
                if sp.issparse(Jg):
                    Jg = Jg.toarray()
                Jg = np.atleast_2d(Jg)
                if pos.size > 0:
                    self._patch_cached.data[pos] = -Jg.ravel()
            return self._patch_cached

        # Cold path: build from scratch
        rows_list, cols_list, vals_list = [], [], []

        for blk in self._blocks:
            q_idx = np.arange(*blk.q_slice.indices(n))
            y_idx = np.arange(*blk.y_slice.indices(n))
            n_qk, n_yk = q_idx.size, y_idx.size

            # Identity on constraint diagonal: q_i, q_i → +1
            if n_qk > 0:
                rows_list.append(q_idx)
                cols_list.append(q_idx)
                vals_list.append(np.ones(n_qk))

            # Negative constraint Jacobian: q_i, y_j → -dg/dy
            if n_qk > 0 and n_yk > 0:
                Jg = blk.call_dg(y[blk.y_slice], t=t, Fk_val=Fk_val)
                if sp.issparse(Jg):
                    Jg = Jg.toarray()
                Jg = np.atleast_2d(Jg)
                rows_list.append(np.repeat(q_idx, n_yk))
                cols_list.append(np.tile(y_idx, n_qk))
                vals_list.append(-Jg.ravel().copy())

        all_r = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
        all_c = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)
        all_v = np.concatenate(vals_list) if vals_list else np.array([], dtype=float)

        patch = sp.coo_matrix((all_v, (all_r, all_c)), shape=(n, n)).tocsr()
        patch.sort_indices()

        # Build index maps for hot-path updates
        # Vectorised: one np.searchsorted call per row (not per element).
        self._patch_dg_positions = []
        for blk in self._blocks:
            q_idx = np.arange(*blk.q_slice.indices(n))
            y_idx = np.arange(*blk.y_slice.indices(n))
            n_qk, n_yk = q_idx.size, y_idx.size
            if n_qk > 0 and n_yk > 0:
                dg_pos = np.empty(n_qk * n_yk, dtype=np.intp)
                for k, qi in enumerate(q_idx):
                    rs = patch.indptr[qi]
                    row_cols = patch.indices[rs:patch.indptr[qi + 1]]
                    dg_pos[k * n_yk:(k + 1) * n_yk] = rs + np.searchsorted(row_cols, y_idx)
                self._patch_dg_positions.append(dg_pos)
            else:
                self._patch_dg_positions.append(np.array([], dtype=np.intp))

        self._patch_cached = patch
        self._patch_n = n
        return patch

    def constraint_residual(self, y, t=None, Fk_val=None, **kwargs):
        r"""Compute the constraint violation vector.

        Returns a vector *r* of length ``len(y)`` where:

        * ``r[field_DOFs] = 0``
        * ``r[q_slice] = y[q_slice] - g(y[y_slice])``

        This is used by the direct Newton solver as the constraint
        portion of the combined residual.

        Parameters
        ----------
        **kwargs : dict
            Extra context forwarded by the solver (e.g. ``step_size``,
            ``prev_state``).  Ignored by the base class but available
            to rate-dependent subclasses.
        """
        r = np.zeros_like(y)
        for blk in self._blocks:
            g_val = blk.call_g(y[blk.y_slice], t=t, Fk_val=Fk_val)
            r[blk.q_slice] = y[blk.q_slice] - g_val
        return r

    def project_batch(self, current_state, candidates, rhok=None, t=None, Fk_val=None):
        C = np.asarray(candidates, dtype=float)
        if C.ndim == 1:
            return self.project(current_state, C, rhok=rhok, t=t, Fk_val=Fk_val)
        out = C.copy()
        for i in range(out.shape[0]):
            for blk in self._blocks:
                y_sub = out[i, blk.y_slice]
                out[i, blk.q_slice] = blk.call_g(y_sub, t=t, Fk_val=Fk_val)
        return out


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
        # Cache the call signature of con_force_func to avoid try/except on every call
        self._con_force_nargs = None  # will be set on first call

    def _call_con_force(self, y, t=None, Fk_val=None):
        # On first call, determine the number of args accepted and cache it
        nargs = self._con_force_nargs
        if nargs == 3:
            return self.con_force_func(y, t, Fk_val)
        elif nargs == 2:
            return self.con_force_func(y, t)
        elif nargs == 1:
            return self.con_force_func(y)
        # First call: probe signature
        try:
            result = self.con_force_func(y, t, Fk_val)
            self._con_force_nargs = 3
            return result
        except TypeError:
            try:
                result = self.con_force_func(y, t)
                self._con_force_nargs = 2
                return result
            except TypeError:
                result = self.con_force_func(y)
                self._con_force_nargs = 1
                return result

    def _compute_jacobian(self, y, t=None, Fk_val=None, rows=None):
        """Compute the constraint force Jacobian.
        
        Parameters
        ----------
        rows : array-like of int, optional
            If provided, only compute these rows of the Jacobian. Returns an
            (len(rows), n) array instead of the full (n, n) Jacobian.
        """
        if self.jac_func is not None:
            try:
                J = self.jac_func(y, t, Fk_val)
            except TypeError:
                J = self.jac_func(y)
            J = _to_numpy_if_needed(J)
            if rows is not None:
                rows = np.asarray(rows)
                if sp.issparse(J):
                    return J[rows].toarray() if hasattr(J[rows], 'toarray') else np.asarray(J[rows])
                return J[rows]
            return J
        return self._numerical_jacobian(y, t=t, Fk_val=Fk_val, rows=rows)

    def _numerical_jacobian(self, y, eps=1e-8, t=None, Fk_val=None, rows=None):
        """Compute the constraint force Jacobian via finite differences.
        
        Parameters
        ----------
        rows : array-like of int, optional
            If provided, only store these rows. Returns (len(rows), n) array.
            When the constraint force Jacobian is diagonal (local constraint
            force), only ``len(rows)`` perturbations are needed instead of n.
        """
        y_np = _to_numpy_if_needed(y)
        n = len(y_np)
        f0 = _to_numpy_if_needed(self._call_con_force(y, t=t, Fk_val=Fk_val))

        if rows is not None:
            rows = np.asarray(rows)
            nr = rows.size

            # Check cached diagonal flag; None = unknown, True/False = detected
            _diag = getattr(self, '_jac_is_diagonal', None)

            if _diag is True:
                # Fast diagonal path: only perturb rows columns
                J = np.zeros((nr, n), dtype=y_np.dtype)
                for idx in range(nr):
                    j = int(rows[idx])
                    y_pert = y_np.copy()
                    y_pert[j] += eps
                    f_eps = _to_numpy_if_needed(self._call_con_force(y_pert, t=t, Fk_val=Fk_val))
                    J[:, j] = (f_eps[rows] - f0[rows]) / eps
                return J

            # Full perturbation (unknown or non-diagonal)
            J = np.zeros((nr, n), dtype=y_np.dtype)
            for j in range(n):
                y_pert = y_np.copy()
                y_pert[j] += eps
                f_eps = _to_numpy_if_needed(self._call_con_force(y_pert, t=t, Fk_val=Fk_val))
                J[:, j] = (f_eps[rows] - f0[rows]) / eps

            if _diag is None:
                # Detect diagonal structure from the computed Jacobian
                off_diag = J.copy()
                for idx in range(nr):
                    off_diag[idx, int(rows[idx])] = 0.0
                self._jac_is_diagonal = bool(np.max(np.abs(off_diag)) < eps * 10)
            return J
        else:
            J = np.zeros((n, n), dtype=y_np.dtype)
            for j in range(n):
                y_pert = y_np.copy()
                y_pert[j] += eps
                f_eps = _to_numpy_if_needed(self._call_con_force(y_pert, t=t, Fk_val=Fk_val))
                J[:, j] = (f_eps - f0) / eps
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
        # Defer Jacobian computation until we know which rows actually need it
        J_conf = None  # will be computed lazily below
        J_conf_rows = None  # row-indexed version: {idx -> row_vector}

        # Helper 2x2 projector matrices (only P00, P01 values matter for top row)
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
                codes = np.asarray(classify_regions_nb(v_arr, zt_arr, tol), dtype=int)
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

        # ---- Fully-vectorized CSR assembly ----
        # P[0,0] and P[0,1] coefficients per code, looked up all at once
        _P00 = np.array([0.0, 1.0, 0.5, 0.5, 0.5, 1.0], dtype=float)  # codes 0-5
        _P01 = np.array([0.0, 0.0, 0.5, -0.5, 0.0, 0.0], dtype=float)
        p00_vec = _P00[codes]  # shape (num_valid,)
        p01_vec = _P01[codes]

        sign_vals = np.sign(current_state[valid_indices]).astype(float)
        sign_vals[sign_vals == 0.0] = 1.0  # convention at the kink

        # Lazily compute the constraint Jacobian only for rows that need P01 != 0
        # (codes 2 and 3), and only compute those specific rows
        _needs_jac_mask = (codes == 2) | (codes == 3)
        _has_jac_rows = self.conf_jacobian_mode == 'full' and np.any(_needs_jac_mask)
        if _has_jac_rows:
            _jac_indices = valid_indices[_needs_jac_mask]
            _jac_rows = self._compute_jacobian(current_state, t=t, Fk_val=Fk_val, rows=_jac_indices)

        # ---- Build the result as a modified identity ----
        # Start with an identity matrix, then modify constrained rows.
        # This avoids a Python for-loop over all n rows.

        # Diagonal values: 1.0 everywhere, then overwritten for constrained rows
        diag_vals = np.ones(n, dtype=float)
        # For constrained rows: diagonal = c0 + c1 * sign  (when c1 != 0)
        #                       diagonal = c0              (when c1 == 0)
        #                       diagonal = 0               (code 0 = zero row)
        diag_vals[valid_indices] = p00_vec + p01_vec * sign_vals

        if not _has_jac_rows:
            # No Jacobian contribution — result is a diagonal matrix
            D_csr = sp.diags(diag_vals, 0, shape=(n, n), format='csr')
        else:
            # Some rows have off-diagonal entries from the Jacobian.
            n_jac = len(_jac_indices)

            # Position of _jac_indices within valid_indices
            _jac_pos = np.searchsorted(valid_indices, _jac_indices)
            c1_jac = p01_vec[_jac_pos]  # c1 values for these rows
            rhok_jac = rhok_full[_jac_indices]  # rhok for these rows
            # Scale factors for Jacobian rows: -c1 * rhok, shape (n_jac,)
            scale = -c1_jac * rhok_jac

            _jac_diag = getattr(self, '_jac_is_diagonal', None)
            if _jac_diag is True:
                # Diagonal constraint Jacobian: correction is purely diagonal
                # jac_rows[k, _jac_indices[k]] is the only nonzero per row
                jac_diag_vals = np.array([
                    float(_jac_rows[k, int(_jac_indices[k])])
                    for k in range(n_jac)])
                diag_vals[_jac_indices] += scale * jac_diag_vals
                D_csr = sp.diags(diag_vals, 0, shape=(n, n), format='csr')
            else:
                # General case: build sparse correction from the dense Jacobian block
                jac_block = _to_numpy_if_needed(_jac_rows) * scale[:, np.newaxis]
                D_csr = sp.diags(diag_vals, 0, shape=(n, n), format='csr')

                # Build COO correction in the full (n, n) space directly
                nz_tol = 1e-15
                jac_block[np.abs(jac_block) <= nz_tol] = 0.0
                nz_rows_local, nz_cols = np.nonzero(jac_block)
                if nz_cols.size > 0:
                    nz_vals = jac_block[nz_rows_local, nz_cols]
                    nz_rows_global = _jac_indices[nz_rows_local]
                    correction = sp.csr_matrix(
                        (nz_vals, (nz_rows_global, nz_cols)),
                        shape=(n, n))
                    D_csr = D_csr + correction

        return D_csr


#########################  #####################################################
# MuScaledSOCProjection — purely geometric μ-scaled SOC projector
##############################################################################
class MuScaledSOCProjection(Projection):
    r"""Purely geometric projector onto the μ-scaled second-order cone.

    For each *block* the projector enforces

    .. math::
        (s, \mathbf{w}) \in K_\mu
        = \bigl\{(s,\mathbf{w}) : s \ge 0,\;
          \|\mathbf{w}\| \le \mu\, s \bigr\}

    where *s* is the **normal component** and *w* the **tangential
    components** (1-D in 2-D contact, 2-D in 3-D contact).  The cone
    simultaneously encodes Signorini (:math:`s \ge 0`) and Coulomb
    friction (:math:`\|\mathbf{w}\| \le \mu s`).

    This class is **physics-agnostic**: the entries may represent
    velocities, impulses, forces, or any other quantity.  Moreau
    time-stepping, De Saxcé transforms, or restitution are handled by
    subclasses that wrap forward/inverse transforms around this
    projector.

    Parameters
    ----------
    blocks : list of (int, array-like of int)
        Each entry ``(idx_N, [idx_T1, ...])`` gives the *state-vector
        index* of the normal component and one or more tangential
        indices.  2-D contact: ``(idx_N, [idx_T])``.
        3-D contact: ``(idx_N, [idx_T1, idx_T2])``.
    get_mu : callable
        Friction coefficient.  Signature ``(y)`` or ``(y, t)`` or
        ``(y, t, Fk_val)`` — arity is auto-detected once via
        ``inspect.signature``.  Must return a scalar (broadcast to all
        blocks) or an array of length ``len(blocks)``.
    gap_func : callable or None, optional
        Signed-gap function ``gap_func(y, t) -> array(n_blocks)``.
        Block *k* is **active** when ``gap[k] <= gap_tol``.  If
        ``None`` (default), all blocks are always active.
    gap_tol : float, default 0.0
        Tolerance for gap activation.
    zero_inactive : bool, default False
        Controls what happens to **inactive** blocks (gap > gap_tol).

        - ``False`` (default, velocity-level): inactive blocks are
          left untouched (identity projection).  Suitable when the
          reaction is implicit in the residual and absence of contact
          means no projection is needed.
        - ``True`` (force/impulse-level): inactive blocks are
          projected onto :math:`\{0\}`, i.e. forced to zero.
          Suitable when the reaction is an explicit DOF in the
          augmented state and free flight requires :math:`\lambda=0`.
          The tangent Jacobian for inactive blocks becomes the zero
          matrix (not identity).
    component_slices : list of slice, optional
        Forwarded to the base ``Projection`` for compatibility with
        the solver's ``_bind_projector_fastpaths``.

    Notes
    -----
    ``rhok`` is accepted by ``project`` / ``tangent_cone`` for
    interface compatibility but is not used by this projector.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, *, blocks, get_mu, gap_func=None,
                 gap_tol=0.0, zero_inactive=False,
                 get_s0=None, get_w0=None,
                 get_ds0_dz=None, get_dw0_dz=None,
                 component_slices=None):
        super().__init__(component_slices=component_slices)

        # Validate and normalize blocks -> list of (int, ndarray[int])
        self.blocks = self._normalize_blocks(blocks)
        self.get_mu = get_mu
        self.gap_func = gap_func
        self.gap_tol = float(gap_tol)
        self.zero_inactive = bool(zero_inactive)

        # Precompute flat index arrays for fast gather / scatter
        self._all_block_indices = []
        for s_idx, w_idx in self.blocks:
            self._all_block_indices.append(np.r_[s_idx, w_idx])

        # Validate block index disjointness (catch overlapping blocks)
        self._validate_block_disjointness()

        # Detect get_mu arity once
        self._mu_nargs = _count_required_args(get_mu)
        # Cache gap_func arity once (avoid inspect.signature on every call)
        self._gap_nargs = _count_required_args(gap_func)

        # Active-set locking (used by SSN to prevent chattering)
        self._locked_active = None   # None = unlocked, ndarray = locked mask

        # Spectral branch persistence:
        # Cache the region index and tangent direction per block to prevent
        # Jacobian oscillation near region boundaries and the cone apex.
        self._branch_region = {}     # block_key -> 0=interior, 1=polar, 2=boundary
        self._branch_w_hat = {}      # block_key -> last non-degenerate unit tangent
        # Scale-aware hysteresis tolerances:
        #   τ = atol + rtol · max(1, |s|, r/μ, μr)
        self._spectral_atol = 1e-14
        self._spectral_rtol = 1e-12

        # Pre-stress callbacks (constant offset of the cone)
        self.get_s0 = get_s0
        self.get_w0 = get_w0
        self._s0_nargs = _count_required_args(get_s0)
        self._w0_nargs = _count_required_args(get_w0)

        # Pre-stress Jacobian callbacks (state-dependent offset)
        self.get_ds0_dz = get_ds0_dz
        self.get_dw0_dz = get_dw0_dz
        self._ds0_dz_nargs = _count_required_args(get_ds0_dz)
        self._dw0_dz_nargs = _count_required_args(get_dw0_dz)

        # ── Batch precomputation for uniform-dimension blocks ────────
        # When all blocks share the same tangential dimension *m* we can
        # vectorize project() and tangent_cone() to avoid Python loops
        # over individual blocks.  This gives order-of-magnitude speedups
        # for contact problems with many identical contacts (e.g. GBK
        # chain, fault mechanics, granular media).
        tang_dims = [len(w_idx) for _, w_idx in self.blocks]
        nb = len(self.blocks)
        if nb > 0 and all(d_k == tang_dims[0] for d_k in tang_dims):
            m = tang_dims[0]
            d = 1 + m
            self._uniform_m = m
            self._batch_s_idx = np.array(
                [s for s, _ in self.blocks], dtype=int)
            self._batch_w_idx = np.array(
                [list(w) for _, w in self.blocks], dtype=int)   # (nb, m)
            # Flat set of all block indices (for non-block row detection)
            self._batch_all_flat = np.concatenate(self._all_block_indices)
            self._batch_non_block_rows = None  # lazily computed
            # Hysteresis state as arrays (mirror the dict-based caches)
            self._batch_branch_region = np.full(nb, -1, dtype=int)
            self._batch_branch_w_hat = np.zeros((nb, m))
            if m > 0:
                self._batch_branch_w_hat[:, 0] = 1.0  # default direction
        else:
            self._uniform_m = None

    # ------------------------------------------------------------------
    # Block validation
    # ------------------------------------------------------------------
    def _validate_block_disjointness(self):
        """Raise ``ValueError`` if any state index appears in more than one block."""
        seen = set()
        for k, idx_arr in enumerate(self._all_block_indices):
            for i in idx_arr:
                i_int = int(i)
                if i_int in seen:
                    raise ValueError(
                        f"State index {i_int} appears in more than one SOC block. "
                        f"Blocks must be disjoint.")
                seen.add(i_int)

    # ------------------------------------------------------------------
    # Block normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_blocks(blocks):
        """Convert user-supplied block specs to a canonical list.

        Accepted forms per element:
        * ``(s_index, [w1, ...])``    — tuple / list
        * ``slice(start, stop)``      — first index = s, remainder = w
        """
        if blocks is None:
            raise ValueError("blocks must be provided (list of (idx_N, [idx_T, ...]))")
        norm = []
        for blk in blocks:
            if isinstance(blk, slice):
                idx = np.arange(blk.start, blk.stop)
                if idx.size < 2:
                    raise ValueError(
                        "SOC block slice must cover >= 2 indices (s + at least 1 w)")
                norm.append((int(idx[0]), np.asarray(idx[1:], dtype=int)))
            else:
                s_idx, w_idx = blk
                w_idx = np.atleast_1d(np.asarray(w_idx, dtype=int))
                if w_idx.size < 1:
                    raise ValueError(
                        "SOC block must include at least one tangential index")
                norm.append((int(s_idx), w_idx))
        return norm

    # ------------------------------------------------------------------
    # μ evaluation with auto-arity
    # ------------------------------------------------------------------
    def _eval_mu(self, y, t=None, Fk_val=None):
        """Call ``get_mu`` with the right number of arguments.

        Returns an ndarray of shape ``(n_blocks,)``; a scalar return
        is broadcast.
        """
        nargs = self._mu_nargs
        if nargs is None or nargs >= 3:
            try:
                mu_val = self.get_mu(y, t, Fk_val)
            except TypeError:
                mu_val = self.get_mu(y, t) if nargs != 1 else self.get_mu(y)
        elif nargs == 2:
            mu_val = self.get_mu(y, t)
        else:
            mu_val = self.get_mu(y)

        mu_arr = np.atleast_1d(np.asarray(mu_val, dtype=float))
        nb = len(self.blocks)
        if mu_arr.size == 1:
            mu_arr = np.full(nb, float(mu_arr.flat[0]))
        elif mu_arr.size == nb:
            mu_arr = mu_arr.ravel()
        else:
            raise ValueError(
                f"get_mu must return a scalar or array of length {nb} "
                f"(got size {mu_arr.size})")
        # Validate: μ must be finite and non-negative
        if not np.all(np.isfinite(mu_arr)):
            raise ValueError(
                f"get_mu returned non-finite value(s): {mu_arr}")
        if np.any(mu_arr < 0.0):
            raise ValueError(
                f"get_mu returned negative value(s): {mu_arr}. "
                f"Friction coefficient must be >= 0.")
        return mu_arr

    # ------------------------------------------------------------------
    # Pre-stress evaluation
    # ------------------------------------------------------------------
    def _eval_s0(self, y, t=None, Fk_val=None):
        """Evaluate normal pre-stress per block.

        Returns an ndarray of shape ``(n_blocks,)``; zeros when
        ``get_s0`` is ``None``.

        Signature of ``get_s0``: ``(y)`` or ``(y, t)`` or
        ``(y, t, Fk_val)`` — same auto-arity as ``get_mu``.
        Must return a scalar (broadcast) or array of length
        ``len(blocks)``.
        """
        if self.get_s0 is None:
            return np.zeros(len(self.blocks))
        nargs = self._s0_nargs
        if nargs is None or nargs >= 3:
            try:
                val = self.get_s0(y, t, Fk_val)
            except TypeError:
                val = self.get_s0(y, t) if nargs != 1 else self.get_s0(y)
        elif nargs == 2:
            val = self.get_s0(y, t)
        else:
            val = self.get_s0(y)
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        nb = len(self.blocks)
        if arr.size == 1:
            arr = np.full(nb, float(arr.flat[0]))
        elif arr.size == nb:
            arr = arr.ravel()
        else:
            raise ValueError(
                f"get_s0 must return a scalar or array of length {nb} "
                f"(got size {arr.size})")
        return arr

    def _eval_w0(self, y, k, t=None, Fk_val=None):
        r"""Evaluate tangential pre-stress for block *k*.

        Parameters
        ----------
        y : ndarray
            Current state vector.
        k : int
            Block index.
        t : float or None
            Current time.
        Fk_val : ndarray or None
            Forcing values.

        Returns
        -------
        w0 : ndarray, shape ``(m_k,)``
            Tangential pre-stress for block *k*; zeros when
            ``get_w0`` is ``None``.

        Notes
        -----
        Signature of ``get_w0``: ``(y, k)`` or ``(y, k, t)`` or
        ``(y, k, t, Fk_val)`` where *k* is the block index.
        Must return an array of length ``m_k`` (tangential
        dimension of block *k*).
        """
        _, w_idx = self.blocks[k]
        m_k = len(w_idx)
        if self.get_w0 is None:
            return np.zeros(m_k)
        nargs = self._w0_nargs
        if nargs is None or nargs >= 4:
            try:
                val = self.get_w0(y, k, t, Fk_val)
            except TypeError:
                try:
                    val = self.get_w0(y, k, t)
                except TypeError:
                    val = self.get_w0(y, k)
        elif nargs == 3:
            val = self.get_w0(y, k, t)
        else:
            val = self.get_w0(y, k)
        w0 = np.atleast_1d(np.asarray(val, dtype=float))
        if w0.size != m_k:
            raise ValueError(
                f"get_w0 must return an array of length {m_k} for block {k} "
                f"(got size {w0.size})")
        return w0

    # ------------------------------------------------------------------
    # Pre-stress Jacobian evaluation (state-dependent offset)
    # ------------------------------------------------------------------
    def _eval_ds0_dz(self, y, t=None, Fk_val=None):
        r"""Evaluate the Jacobian of *s0* w.r.t. the state vector.

        Returns ``(n_blocks, n)`` array, or ``None`` when
        ``get_ds0_dz`` is not configured.

        ``get_ds0_dz`` may return:

        * ``(n,)``       — single gradient, broadcast to all blocks,
        * ``(1, n)``     — same, but 2-D,
        * ``(n_blocks, n)`` — per-block gradients.

        Signature follows ``get_s0``: ``(y)`` or ``(y, t)`` or
        ``(y, t, Fk_val)``.
        """
        if self.get_ds0_dz is None:
            return None
        nargs = self._ds0_dz_nargs
        if nargs is None or nargs >= 3:
            try:
                val = self.get_ds0_dz(y, t, Fk_val)
            except TypeError:
                val = (self.get_ds0_dz(y, t) if nargs != 1
                       else self.get_ds0_dz(y))
        elif nargs == 2:
            val = self.get_ds0_dz(y, t)
        else:
            val = self.get_ds0_dz(y)
        arr = np.asarray(val, dtype=float)
        nb = len(self.blocks)
        n = y.size
        if arr.ndim == 1:
            if arr.size != n:
                raise ValueError(
                    f"get_ds0_dz returned 1-D array of length {arr.size}, "
                    f"expected {n} (state dimension)")
            arr = np.tile(arr.reshape(1, -1), (nb, 1))
        elif arr.shape == (1, n):
            arr = np.tile(arr, (nb, 1))
        elif arr.shape != (nb, n):
            raise ValueError(
                f"get_ds0_dz must return shape ({nb}, {n}) or ({n},), "
                f"got {arr.shape}")
        return arr

    def _eval_dw0_dz(self, y, k, t=None, Fk_val=None):
        r"""Evaluate Jacobian of *w0* for block *k* w.r.t. the state.

        Returns ``(m_k, n)`` array, or ``None`` when
        ``get_dw0_dz`` is not configured.

        Signature follows ``get_w0``: ``(y, k)`` or ``(y, k, t)``
        or ``(y, k, t, Fk_val)``.
        """
        if self.get_dw0_dz is None:
            return None
        _, w_idx = self.blocks[k]
        m_k = len(w_idx)
        n = y.size
        nargs = self._dw0_dz_nargs
        if nargs is None or nargs >= 4:
            try:
                val = self.get_dw0_dz(y, k, t, Fk_val)
            except TypeError:
                try:
                    val = self.get_dw0_dz(y, k, t)
                except TypeError:
                    val = self.get_dw0_dz(y, k)
        elif nargs == 3:
            val = self.get_dw0_dz(y, k, t)
        else:
            val = self.get_dw0_dz(y, k)
        arr = np.asarray(val, dtype=float)
        if arr.shape != (m_k, n):
            raise ValueError(
                f"get_dw0_dz must return shape ({m_k}, {n}) for block {k}, "
                f"got {arr.shape}")
        return arr

    def _assemble_dz0_block(self, y, k, t, Fk_val, n, ds0_dz_all):
        r"""Assemble the ``(d_k, n)`` Jacobian of :math:`z_0^{(k)}` w.r.t. the state.

        Parameters
        ----------
        ds0_dz_all : ndarray (n_blocks, n) or None
            Pre-computed result of :meth:`_eval_ds0_dz` (avoids
            redundant evaluation across blocks).

        Returns
        -------
        dz0 : ndarray (d_k, n) or None
            ``None`` when neither ``get_ds0_dz`` nor ``get_dw0_dz``
            is configured.
        """
        _has = self.get_ds0_dz is not None or self.get_dw0_dz is not None
        if not _has:
            return None
        _, w_idx = self.blocks[k]
        m_k = len(w_idx)
        d_k = 1 + m_k
        dz0 = np.zeros((d_k, n))
        if ds0_dz_all is not None:
            dz0[0, :] = ds0_dz_all[k, :]
        if self.get_dw0_dz is not None:
            dw0 = self._eval_dw0_dz(y, k, t, Fk_val)
            if dw0 is not None:
                dz0[1:, :] = dw0
        return dz0

    # ------------------------------------------------------------------
    # Core static projector  (s, w) -> Π_{K_μ}(s, w)
    # ------------------------------------------------------------------
    @staticmethod
    def _proj_mu_scaled_soc(z, mu, return_jacobian=False, eps=1e-30):
        r"""Project ``z = (s, w)`` onto :math:`K_\mu` via spectral eigenvalues.

        Spectral eigenvalues of the μ-scaled SOC:

        .. math::
            \lambda_+ = s + \mu\,r, \qquad
            \lambda_- = s - r/\mu \quad (\mu > 0)

        where :math:`r = \|w\|`.  Region classification:

        * **Interior**: :math:`\lambda_- \ge 0` and :math:`s \ge 0`
        * **Polar**: :math:`\lambda_+ \le 0`
        * **Boundary**: :math:`\lambda_+ > 0` and :math:`\lambda_- < 0`

        Projection via positive-part of eigenvalues:

        .. math::
            \Pi_{K_\mu}(z) = (\lambda_+)_+\, c_+ + (\lambda_-)_+\, c_-

        No value regularisation is needed: when :math:`r = 0`,
        :math:`\lambda_+ = \lambda_- = s`, so the point is interior
        (:math:`s \ge 0`) or polar (:math:`s \le 0`), never boundary.

        Parameters
        ----------
        z : ndarray, shape (1+m,)
            The vector ``[s, w_1, ..., w_m]``.
        mu : float
            Friction coefficient (cone opening).
        return_jacobian : bool
            If ``True`` return ``(projection, jacobian)`` where
            *jacobian* is the ``(1+m, 1+m)`` Clarke sub-differential.
        eps : float
            Kept for API compatibility; only used to stabilise the
            angular-stiffness ratio :math:`\lambda_+/r` in the Jacobian
            when :math:`r \to 0`.

        Returns
        -------
        p : ndarray, shape (1+m,)
            or ``(p, J)`` when ``return_jacobian=True``.
        """
        z = np.asarray(z, dtype=float)
        s = float(z[0])
        w = z[1:].copy()
        m = w.size
        d = 1 + m                       # block dimension
        r = float(np.linalg.norm(w))     # ||w||

        # ---- μ = 0: degenerate cone K_0 = {(s, 0) : s ≥ 0} ----
        if mu <= 0.0:
            p = np.zeros(d)
            p[0] = max(s, 0.0)
            if not return_jacobian:
                return p
            J = np.zeros((d, d))
            if s >= 0.0:
                J[0, 0] = 1.0       # Clarke selection at s = 0
            return p, J

        # ---- Spectral eigenvalues ----
        lam_plus  = s + mu * r       # λ₊ = s + μ‖w‖
        lam_minus = s - r / mu       # λ₋ = s − ‖w‖/μ

        # ---- Region 1: interior (λ₋ ≥ 0 and s ≥ 0) ----
        if lam_minus >= 0.0 and s >= 0.0:
            if return_jacobian:
                return z.copy(), np.eye(d)
            return z.copy()

        # ---- Region 2: polar (λ₊ ≤ 0) ----
        if lam_plus <= 0.0:
            if return_jacobian:
                return np.zeros(d), np.zeros((d, d))
            return np.zeros(d)

        # ---- Region 3: boundary (λ₊ > 0, λ₋ < 0) ----
        # r > 0 is guaranteed here: λ₊ > 0 and λ₋ < 0 with μ > 0
        # implies r > 0 (if r = 0, then λ₊ = λ₋ = s, same sign).
        alpha = 1.0 / (1.0 + mu * mu)
        w_hat = w / r                  # exact, no regularisation

        p = np.empty(d)
        p[0] = alpha * lam_plus
        p[1:] = (alpha * mu * lam_plus) * w_hat

        if not return_jacobian:
            return p

        # Clarke sub-differential on the boundary:
        #   J = α [ 1          μ ŵᵀ                         ]
        #         [ μ ŵ        μ² ŵŵᵀ + μ(λ₊/r)(I − ŵŵᵀ)  ]
        wwT = np.outer(w_hat, w_hat)
        r_jac = max(r, eps)            # stabilise λ₊/r near apex
        J = np.empty((d, d))
        J[0, 0] = alpha
        J[0, 1:] = alpha * mu * w_hat
        J[1:, 0] = alpha * mu * w_hat
        J[1:, 1:] = alpha * (
            mu * mu * wwT
            + mu * (lam_plus / r_jac) * (np.eye(m) - wwT)
        )

        return p, J

    # ------------------------------------------------------------------
    # Spectral-consistent projector with branch persistence
    # ------------------------------------------------------------------
    def _proj_persistent(self, z, mu, block_key, return_jacobian=False):
        r"""Spectral projector with branch persistence for **Jacobian only**.

        * When ``return_jacobian=False``: delegates to
          :meth:`_proj_mu_scaled_soc` for an **exact** projection value
          with no hysteresis or cached directions.  This guarantees that
          the solver always evaluates the true operator
          :math:`\Pi_{K_\mu}`.

        * When ``return_jacobian=True``: uses scale-aware hysteresis on
          the spectral eigenvalues :math:`\lambda_\pm` and a cached
          tangent direction :math:`\hat w` near the apex to select a
          **stable Clarke element** for semismooth Newton Jacobians.
          The projection **value** returned alongside the Jacobian is
          still exact (computed without hysteresis).

        Spectral eigenvalues of the μ-scaled SOC
        :math:`K_\mu = \{(s,w): s \ge 0,\; \|w\| \le \mu s\}`:

        .. math::
            \lambda_+ = s + \mu\,r, \qquad
            \lambda_- = s - r/\mu \quad (\mu > 0)

        where :math:`r = \|w\|`.  Region classification:

        * **Interior**: :math:`\lambda_- \ge 0` and :math:`s \ge 0`
        * **Polar**: :math:`\lambda_+ \le 0`
        * **Boundary**: :math:`\lambda_+ > 0` and :math:`\lambda_- < 0`

        Parameters
        ----------
        z : ndarray, shape (1+m,)
        mu : float
        block_key : hashable
            Cache key (typically the block index *k*).
        return_jacobian : bool
        """
        # ---- Exact value path (no persistence) ----
        if not return_jacobian:
            return self._proj_mu_scaled_soc(z, mu, return_jacobian=False)

        # ---- Jacobian path: hysteresis + cached ŵ ----
        z = np.asarray(z, dtype=float)
        s = float(z[0])
        w = z[1:]
        m = w.size
        d = 1 + m
        r = float(np.linalg.norm(w))

        # Exact projection value (always returned without hysteresis)
        p = self._proj_mu_scaled_soc(z, mu, return_jacobian=False)

        # ---- μ = 0: degenerate cone K_0 = {(s, 0) : s ≥ 0} ----
        if mu <= 0.0:
            J = np.zeros((d, d))
            if s >= 0.0:
                J[0, 0] = 1.0  # deterministic Clarke element at s=0
            return p, J

        # ---- Spectral eigenvalues ----
        lam_plus  = s + mu * r       # λ₊ = s + μ‖w‖
        lam_minus = s - r / mu       # λ₋ = s - ‖w‖/μ

        # ---- Scale-aware tolerance ----
        atol = self._spectral_atol
        rtol = self._spectral_rtol
        scale = max(1.0, abs(s), r / mu, mu * r)
        tau = atol + rtol * scale

        # ---- Classify region from eigenvalue signs ----
        if lam_minus >= 0.0 and s >= 0.0:
            region = 0  # interior
        elif lam_plus <= 0.0:
            region = 1  # polar
        else:
            region = 2  # boundary

        # ---- Hysteresis: persist previous region near λ± = 0 ----
        prev_region = self._branch_region.get(block_key)
        if prev_region is not None and region != prev_region:
            if abs(lam_minus) < tau or abs(lam_plus) < tau:
                region = prev_region
        self._branch_region[block_key] = region

        # ---- Interior Jacobian (I) ----
        if region == 0:
            return p, np.eye(d)

        # ---- Polar Jacobian (0) ----
        if region == 1:
            return p, np.zeros((d, d))

        # ---- Boundary Jacobian ----
        alpha = 1.0 / (1.0 + mu * mu)

        # Tangent direction ŵ:
        # r > 0 is guaranteed on the true boundary (no hysteresis).
        # The only way to reach boundary with r ≈ 0 is via hysteresis;
        # use cached ŵ then.
        if r > 0.0:
            w_hat = w / r
            # Update cache only when r is numerically reliable
            if r > tau:
                self._branch_w_hat[block_key] = w_hat.copy()
        else:
            # r = 0 on boundary (hysteresis only) — use cache
            prev_w_hat = self._branch_w_hat.get(block_key)
            if prev_w_hat is not None and prev_w_hat.size == m:
                w_hat = prev_w_hat
            else:
                w_hat = np.zeros(m)
                if m > 0:
                    w_hat[0] = 1.0

        # Clarke subdifferential on boundary:
        # J = α [ 1          μ ŵᵀ                              ]
        #       [ μ ŵ        μ² ŵŵᵀ + μ(λ₊/r)(I − ŵŵᵀ)       ]
        wwT = np.outer(w_hat, w_hat)
        r_jac = max(r, atol)  # stabilise λ₊/r near apex

        J = np.empty((d, d))
        J[0, 0] = alpha
        J[0, 1:] = alpha * mu * w_hat
        J[1:, 0] = alpha * mu * w_hat
        J[1:, 1:] = alpha * (
            mu * mu * wwT
            + mu * (lam_plus / r_jac) * (np.eye(m) - wwT)
        )

        return p, J

    def reset_branch_cache(self):
        """Clear spectral branch persistence caches.

        Called at the start of a new nonlinear solve so that stale
        region / direction information from a previous time step
        does not leak.
        """
        self._branch_region.clear()
        self._branch_w_hat.clear()
        # Also reset batch arrays (vectorized fast-path)
        if self._uniform_m is not None:
            self._batch_branch_region[:] = -1
            m = self._uniform_m
            self._batch_branch_w_hat[:] = 0.0
            if m > 0:
                self._batch_branch_w_hat[:, 0] = 1.0

    # ------------------------------------------------------------------
    # Gap-based activation
    # ------------------------------------------------------------------
    def _active_mask(self, y, t):
        """Return boolean mask of length ``n_blocks`` (True = active)."""
        # If an external solver locked the active set, use it.
        if self._locked_active is not None:
            return self._locked_active
        if self.gap_func is None:
            return np.ones(len(self.blocks), dtype=bool)
        nargs = self._gap_nargs          # cached at __init__ time
        if nargs is not None and nargs <= 1:
            gaps = np.atleast_1d(self.gap_func(y))
        else:
            gaps = np.atleast_1d(self.gap_func(y, t))
        return gaps <= self.gap_tol

    def lock_active_set(self, y, t=None, reset_branch=True):
        """Evaluate and freeze the active-set mask.

        While locked, ``_active_mask`` always returns this mask,
        preventing active-set chattering in Newton-type solvers.
        Call ``unlock_active_set`` when done.

        Parameters
        ----------
        y : ndarray
            State at which to evaluate the gap.
        t : float or None
            Current time.
        reset_branch : bool, default True
            If ``True``, clear spectral branch persistence caches.
            Set to ``False`` for merit-based re-locking within a
            solve (Proposal 3) so that Jacobian branch choices
            are preserved.
        """
        if reset_branch:
            self.reset_branch_cache()
        if self.gap_func is None:
            self._locked_active = np.ones(len(self.blocks), dtype=bool)
            return
        nargs = self._gap_nargs
        if nargs is not None and nargs <= 1:
            gaps = np.atleast_1d(self.gap_func(y))
        else:
            gaps = np.atleast_1d(self.gap_func(y, t))
        self._locked_active = (gaps <= self.gap_tol)

    def unlock_active_set(self):
        """Release the frozen active-set mask and clear branch caches."""
        self._locked_active = None
        self.reset_branch_cache()

    # ------------------------------------------------------------------
    # Projection API
    # ------------------------------------------------------------------
    # Vectorized batch project / tangent_cone for uniform-dim blocks
    # ------------------------------------------------------------------
    def _batch_eligible(self):
        """Check if the vectorized batch fast-path can be used.

        Requirements:
        * All blocks have the same tangential dimension (``_uniform_m``).
        * No tangential pre-stress callback (``get_w0 is None``).
        * No state-dependent pre-stress Jacobian (``get_ds0_dz`` and
          ``get_dw0_dz`` are both ``None``).
        """
        return (self._uniform_m is not None
                and self.get_w0 is None
                and self.get_ds0_dz is None
                and self.get_dw0_dz is None)

    def _batch_project_fast(self, z_work, active, mu_arr, s0_arr):
        """Vectorized projection for uniform-dimension blocks (no w0)."""
        nb = len(self.blocks)
        m = self._uniform_m
        s_idx = self._batch_s_idx
        w_idx = self._batch_w_idx          # (nb, m)

        # Gather: s (nb,) and w (nb, m)
        s = z_work[s_idx]
        w = z_work[w_idx].reshape(nb, m)   # ensure 2-D

        # Apply normal pre-stress shift
        s_shifted = s + s0_arr

        # Tangential norms — one vectorized call
        if m == 1:
            r = np.abs(w[:, 0])
        else:
            r = np.linalg.norm(w, axis=1)

        # Spectral eigenvalues
        mu_safe = np.maximum(mu_arr, 1e-30)
        lam_plus  = s_shifted + mu_arr * r
        lam_minus = s_shifted - r / mu_safe

        # Region masks
        interior = active & (lam_minus >= 0.0) & (s_shifted >= 0.0)
        polar    = active & (lam_plus  <= 0.0)
        boundary = active & ~interior & ~polar
        inactive = ~active

        # Interior blocks: z_proj = z — already in z_work, no-op.

        # Polar blocks: project to zero, then un-shift s0
        if np.any(polar):
            z_work[s_idx[polar]] = -s0_arr[polar]
            z_work[w_idx[polar]] = 0.0

        # Boundary blocks: spectral projection + un-shift
        bnd = np.flatnonzero(boundary)
        if bnd.size > 0:
            alpha = 1.0 / (1.0 + mu_arr[bnd] ** 2)
            lp = lam_plus[bnd]
            r_safe = np.maximum(r[bnd], 1e-30)
            w_bnd = w[bnd]                     # (n_bnd, m)
            w_hat = w_bnd / r_safe[:, None]

            s_proj = alpha * lp - s0_arr[bnd]  # un-shift
            w_proj = (alpha * mu_arr[bnd] * lp)[:, None] * w_hat

            z_work[s_idx[bnd]] = s_proj
            if m == 1:
                z_work[w_idx[bnd, 0]] = w_proj[:, 0]
            else:
                z_work[w_idx[bnd]] = w_proj

        # Inactive blocks with zero_inactive
        if self.zero_inactive and np.any(inactive):
            z_work[s_idx[inactive]] = 0.0
            z_work[w_idx[inactive]] = 0.0

        return z_work

    def _batch_tangent_cone_fast(self, z, y, active, mu_arr, s0_arr, n):
        """Vectorized tangent-cone CSR assembly for uniform-dim blocks."""
        nb = len(self.blocks)
        m = self._uniform_m
        d = 1 + m
        s_idx = self._batch_s_idx
        w_idx = self._batch_w_idx          # (nb, m)

        # Gather block data with pre-stress shift
        s = z[s_idx] + s0_arr
        w = z[w_idx].reshape(nb, m)

        # Tangential norms
        if m == 1:
            r = np.abs(w[:, 0])
        else:
            r = np.linalg.norm(w, axis=1)

        # Spectral eigenvalues
        mu_safe = np.maximum(mu_arr, 1e-30)
        lam_plus  = s + mu_arr * r
        lam_minus = s - r / mu_safe

        # Region classification (0=interior, 1=polar, 2=boundary)
        region = np.full(nb, 2, dtype=int)
        region[(lam_minus >= 0.0) & (s >= 0.0)] = 0
        region[lam_plus <= 0.0] = 1

        # Scale-aware hysteresis (vectorized)
        scale = np.maximum.reduce([np.ones(nb), np.abs(s),
                                   r / mu_safe, mu_arr * r])
        tau = self._spectral_atol + self._spectral_rtol * scale

        prev_region = self._batch_branch_region
        differs  = (prev_region >= 0) & (region != prev_region)
        near_zero = (np.abs(lam_minus) < tau) | (np.abs(lam_plus) < tau)
        hysteresis = differs & near_zero
        region[hysteresis] = prev_region[hysteresis]
        self._batch_branch_region[:] = region

        # Update w_hat cache (boundary blocks only)
        boundary = (region == 2) & active
        bnd = np.flatnonzero(boundary)
        if bnd.size > 0:
            r_bnd = r[bnd]
            r_safe = np.maximum(r_bnd, 1e-30)
            w_bnd = w[bnd]
            w_hat_bnd = w_bnd / r_safe[:, None]
            # Use cached direction for degenerate r
            degen = r_bnd < 1e-14
            if np.any(degen):
                w_hat_bnd[degen] = self._batch_branch_w_hat[bnd[degen]]
            # Update cache where r is reliable
            reliable = r_bnd > tau[bnd]
            if np.any(reliable):
                self._batch_branch_w_hat[bnd[reliable]] = w_hat_bnd[reliable]

        # ── Build CSR via COO arrays ─────────────────────────────────
        # Non-block rows: identity (computed once, cached)
        if self._batch_non_block_rows is None:
            all_block_set = set(self._batch_all_flat.tolist())
            self._batch_non_block_rows = np.array(
                [i for i in range(n) if i not in all_block_set], dtype=int)

        nbr = self._batch_non_block_rows
        # Preallocate COO lists — estimate max size
        max_nnz = len(nbr) + nb * d * d
        rows = np.empty(max_nnz, dtype=int)
        cols = np.empty(max_nnz, dtype=int)
        data = np.empty(max_nnz)
        ptr = 0

        # 1) Non-block identity rows
        nn = len(nbr)
        if nn:
            rows[ptr:ptr+nn] = nbr
            cols[ptr:ptr+nn] = nbr
            data[ptr:ptr+nn] = 1.0
            ptr += nn

        # 2) Interior blocks → identity on block indices
        interior = (region == 0) & active
        int_idx = np.flatnonzero(interior)
        if int_idx.size:
            si = s_idx[int_idx]
            wi = w_idx[int_idx].ravel()
            all_int = np.concatenate([si, wi])
            ni = all_int.size
            rows[ptr:ptr+ni] = all_int
            cols[ptr:ptr+ni] = all_int
            data[ptr:ptr+ni] = 1.0
            ptr += ni

        # 3) Polar blocks → zero rows (skip — no entries)

        # 4) Inactive blocks: identity if not zero_inactive, else zero
        inactive = ~active
        if not self.zero_inactive and np.any(inactive):
            inact_idx = np.flatnonzero(inactive)
            si = s_idx[inact_idx]
            wi = w_idx[inact_idx].ravel()
            all_inact = np.concatenate([si, wi])
            ni = all_inact.size
            rows[ptr:ptr+ni] = all_inact
            cols[ptr:ptr+ni] = all_inact
            data[ptr:ptr+ni] = 1.0
            ptr += ni

        # 5) Boundary blocks → d×d block Jacobians
        if bnd.size > 0:
            alpha = 1.0 / (1.0 + mu_arr[bnd] ** 2)
            mu_bnd = mu_arr[bnd]
            lp = lam_plus[bnd]
            r_jac = np.maximum(r[bnd], self._spectral_atol)

            if m == 1:
                # Optimised m=1 path: 2×2 blocks, 4 entries each
                # J = α [[1, μ·ŵ], [μ·ŵ, μ²]]
                # (for m=1: ŵŵᵀ=1, I−ŵŵᵀ=0)
                wh1 = w_hat_bnd[:, 0]       # (n_bnd,) scalar w_hat
                J00 = alpha
                J01 = alpha * mu_bnd * wh1
                J10 = J01
                J11 = alpha * mu_bnd ** 2

                n_bnd = bnd.size
                sb = s_idx[bnd]
                wb = w_idx[bnd, 0]

                # Row indices: [s, s, w, w] per block
                rows[ptr:ptr+4*n_bnd:4] = sb
                rows[ptr+1:ptr+4*n_bnd:4] = sb
                rows[ptr+2:ptr+4*n_bnd:4] = wb
                rows[ptr+3:ptr+4*n_bnd:4] = wb
                # Col indices: [s, w, s, w] per block
                cols[ptr:ptr+4*n_bnd:4] = sb
                cols[ptr+1:ptr+4*n_bnd:4] = wb
                cols[ptr+2:ptr+4*n_bnd:4] = sb
                cols[ptr+3:ptr+4*n_bnd:4] = wb
                # Values
                data[ptr:ptr+4*n_bnd:4] = J00
                data[ptr+1:ptr+4*n_bnd:4] = J01
                data[ptr+2:ptr+4*n_bnd:4] = J10
                data[ptr+3:ptr+4*n_bnd:4] = J11
                ptr += 4 * n_bnd

            elif m == 2:
                # 3×3 blocks, 9 entries each
                wh = w_hat_bnd                 # (n_bnd, 2)
                n_bnd = bnd.size

                # Block Jacobians (vectorized)
                # J[0,0] = α
                # J[0,1:] = α μ ŵ             → 2 entries
                # J[1:,0] = α μ ŵ             → 2 entries
                # J[1:,1:] = α(μ² ŵŵᵀ + μ(λ₊/r)(I − ŵŵᵀ))  → 4 entries
                aM = alpha * mu_bnd            # (n_bnd,)
                aMM = alpha * mu_bnd ** 2
                aR = alpha * mu_bnd * (lp / r_jac)
                # Tangential 2×2 entries (vectorized outer + isotropic)
                ww00 = wh[:, 0] ** 2;  ww01 = wh[:, 0] * wh[:, 1]
                ww11 = wh[:, 1] ** 2
                T00 = aMM * ww00 + aR * (1.0 - ww00)
                T01 = aMM * ww01 + aR * (0.0 - ww01)
                T10 = T01
                T11 = aMM * ww11 + aR * (1.0 - ww11)

                sb = s_idx[bnd]
                w0b = w_idx[bnd, 0]
                w1b = w_idx[bnd, 1]

                # 9 entries per block → stride-9
                base = ptr
                for j, (r_arr, c_arr, d_arr) in enumerate([
                    (sb, sb, alpha),
                    (sb, w0b, aM * wh[:, 0]),
                    (sb, w1b, aM * wh[:, 1]),
                    (w0b, sb, aM * wh[:, 0]),
                    (w0b, w0b, T00),
                    (w0b, w1b, T01),
                    (w1b, sb, aM * wh[:, 1]),
                    (w1b, w0b, T10),
                    (w1b, w1b, T11),
                ]):
                    rows[base+j::9][:n_bnd] = r_arr
                    cols[base+j::9][:n_bnd] = c_arr
                    data[base+j::9][:n_bnd] = d_arr
                ptr += 9 * n_bnd

            else:
                # General m: fall back to per-block assembly
                for k_loc, k in enumerate(bnd):
                    idx_k = self._all_block_indices[k]
                    wh_k = w_hat_bnd[k_loc]
                    mu_k = mu_bnd[k_loc]
                    a = alpha[k_loc]
                    r_k = r_jac[k_loc]
                    lp_k = lp[k_loc]
                    wwT = np.outer(wh_k, wh_k)
                    J_blk = np.empty((d, d))
                    J_blk[0, 0] = a
                    J_blk[0, 1:] = a * mu_k * wh_k
                    J_blk[1:, 0] = a * mu_k * wh_k
                    J_blk[1:, 1:] = a * (
                        mu_k ** 2 * wwT
                        + mu_k * (lp_k / r_k) * (np.eye(m) - wwT))
                    rr = np.repeat(idx_k, d)
                    cc = np.tile(idx_k, d)
                    ne = d * d
                    rows[ptr:ptr+ne] = rr
                    cols[ptr:ptr+ne] = cc
                    data[ptr:ptr+ne] = J_blk.ravel()
                    ptr += ne

        return sp.csr_matrix(
            (data[:ptr].copy(), (rows[:ptr].copy(), cols[:ptr].copy())),
            shape=(n, n))

    # ------------------------------------------------------------------
    def project(self, current_state, candidate, rhok=None, t=None,
                Fk_val=None, prev_state=None, step_size=None, **kw):
        """Project *candidate* block-wise onto :math:`K_\\mu`.

        Uses the exact projector :meth:`_proj_mu_scaled_soc` (no
        hysteresis) so the value map is always :math:`\\Pi_{K_\\mu}`.
        Inactive blocks (gap > 0) are left untouched.
        """
        y = np.asarray(current_state, dtype=float)
        z_work = np.asarray(candidate, dtype=float).copy()

        active = self._active_mask(y, t)
        mu_arr = self._eval_mu(y, t=t, Fk_val=Fk_val)

        # Pre-stress evaluation (skip when no callbacks configured)
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        # ── Vectorized batch fast-path ──
        if self._batch_eligible():
            _s0 = s0_arr if s0_arr is not None else np.zeros(len(self.blocks))
            return self._batch_project_fast(z_work, active, mu_arr, _s0)

        for k, (s_idx, w_idx) in enumerate(self.blocks):
            idx = self._all_block_indices[k]
            if not active[k]:
                if self.zero_inactive:
                    z_work[idx] = 0.0
                continue
            mu_k = float(mu_arr[k])
            # Gather block vector [s, w1, ..., wm]
            z_blk = z_work[idx]
            # Apply pre-stress shift (translate cone)
            if _has_ps:
                z_blk = z_blk.copy()
                s0_k = float(s0_arr[k])
                w0_k = self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                z_blk[0] += s0_k
                z_blk[1:] += w0_k
            # Exact projection – no persistence (persistence is for
            # Jacobian selection only, via _proj_persistent).
            z_proj = self._proj_mu_scaled_soc(z_blk, mu_k)
            # Un-shift pre-stress
            if _has_ps:
                z_proj[0] -= s0_k
                z_proj[1:] -= w0_k
            z_work[idx] = z_proj

        return z_work

    # ------------------------------------------------------------------
    # Tangent cone (Clarke sub-differential)
    # ------------------------------------------------------------------
    def tangent_cone(self, candidate, current_state, rhok=None, t=None,
                     Fk_val=None, prev_state=None, step_size=None, **kw):
        r"""Assemble the Clarke sub-differential of the block-wise
        SOC projection as a sparse CSR matrix (or dense ndarray for
        small systems).

        Non-block rows are identity.  For each active block the
        corresponding rows are filled with the block Jacobian from
        ``_proj_mu_scaled_soc(..., return_jacobian=True)``.

        When ``get_ds0_dz`` or ``get_dw0_dz`` are provided (state-
        dependent pre-stress), the correction term
        :math:`(J_{\mathrm{cone}} - I)\,\partial z_0 / \partial z`
        is added to the block rows, which may introduce off-diagonal
        entries coupling the block to non-block DOFs.
        """
        y = np.asarray(current_state, dtype=float)
        z = np.asarray(candidate, dtype=float)
        n = z.size

        active = self._active_mask(y, t)
        mu_arr = self._eval_mu(y, t=t, Fk_val=Fk_val)

        # Pre-stress evaluation (skip when no callbacks configured)
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        # Pre-stress *Jacobian* evaluation (state-dependent offset)
        _has_ps_jac = self.get_ds0_dz is not None or self.get_dw0_dz is not None
        ds0_dz_all = (self._eval_ds0_dz(y, t=t, Fk_val=Fk_val)
                      if _has_ps_jac and self.get_ds0_dz is not None
                      else None)

        # ── Vectorized batch fast-path (large systems, uniform blocks) ──
        if n > 64 and self._batch_eligible():
            _s0 = s0_arr if s0_arr is not None else np.zeros(len(self.blocks))
            return self._batch_tangent_cone_fast(z, y, active, mu_arr, _s0, n)

        # --- Dense fast-path for small systems (avoids CSR overhead) ---
        if n <= 64:
            D = np.eye(n)
            for k, (s_idx, w_idx) in enumerate(self.blocks):
                idx = self._all_block_indices[k]
                if not active[k]:
                    if self.zero_inactive:
                        D[np.ix_(idx, idx)] = 0.0
                    continue
                mu_k = float(mu_arr[k])
                z_blk = z[idx]
                # Shift to pre-stressed coordinates for Jacobian evaluation
                if _has_ps:
                    z_blk = z_blk.copy()
                    z_blk[0] += float(s0_arr[k])
                    z_blk[1:] += self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                _, J_blk = self._proj_persistent(z_blk, mu_k,
                                                  block_key=k,
                                                  return_jacobian=True)
                D[np.ix_(idx, idx)] = J_blk
                # State-dependent pre-stress Jacobian correction:
                # D[idx, :] += (J_blk - I) @ dz0_k/dz
                if _has_ps_jac:
                    dz0_k = self._assemble_dz0_block(
                        y, k, t, Fk_val, n, ds0_dz_all)
                    if dz0_k is not None:
                        d_k = J_blk.shape[0]
                        D[idx, :] += (J_blk - np.eye(d_k)) @ dz0_k
            return D

        # --- Sparse path for larger systems ---
        # Pre-compute block Jacobians once per active block
        block_jacs = {}  # k -> (J_blk, dz0_k or None)
        zero_block_indices = set()  # global indices that are zeroed (inactive)
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            idx = self._all_block_indices[k]
            if not active[k]:
                if self.zero_inactive:
                    zero_block_indices.update(int(i) for i in idx)
                continue
            mu_k = float(mu_arr[k])
            z_blk = z[idx]
            # Shift to pre-stressed coordinates for Jacobian evaluation
            if _has_ps:
                z_blk = z_blk.copy()
                z_blk[0] += float(s0_arr[k])
                z_blk[1:] += self._eval_w0(y, k, t=t, Fk_val=Fk_val)
            _, J_blk = self._proj_persistent(z_blk, mu_k,
                                              block_key=k,
                                              return_jacobian=True)
            dz0_k = (self._assemble_dz0_block(y, k, t, Fk_val, n, ds0_dz_all)
                     if _has_ps_jac else None)
            block_jacs[k] = (J_blk, dz0_k)

        # Build a mapping: global_row -> (block_index, local_row, idx_array)
        block_row_map = {}
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            idx = self._all_block_indices[k]
            for local_r, global_r in enumerate(idx):
                block_row_map[int(global_r)] = (k, local_r, idx)

        # Assemble CSR row by row
        data = []
        col_indices = []
        indptr = [0]
        nz_tol = 1e-15

        for r in range(n):
            if r in zero_block_indices:
                # Inactive block with zero_inactive: zero row
                indptr.append(len(data))
                continue
            entry = block_row_map.get(r)
            if entry is None:
                # Identity row
                data.append(1.0)
                col_indices.append(r)
                indptr.append(len(data))
                continue

            k, local_r, idx = entry
            J_blk, dz0_k = block_jacs[k]
            if dz0_k is not None:
                # State-dependent pre-stress: full n-wide row
                d_k = J_blk.shape[0]
                corr = ((J_blk - np.eye(d_k)) @ dz0_k)[local_r, :]
                full_row = np.zeros(n)
                full_row[idx] = J_blk[local_r, :]
                full_row += corr
                nz = np.flatnonzero(np.abs(full_row) > nz_tol)
                if nz.size:
                    data.extend(full_row[nz].tolist())
                    col_indices.extend(nz.tolist())
            else:
                row_vals = J_blk[local_r, :]
                nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
                if nz.size:
                    data.extend(row_vals[nz].tolist())
                    col_indices.extend(idx[nz].tolist())
            indptr.append(len(data))

        return sp.csr_matrix(
            (np.array(data, dtype=float),
             np.array(col_indices, dtype=int),
             np.array(indptr, dtype=int)),
            shape=(n, n))

##############################################################################
# CompositeContactProjection — algebraic constraints + SOC contact
##############################################################################
class CompositeContactProjection(Projection):
    r"""Composite projection combining algebraic constraints with SOC contact.

    Wraps an :class:`AlgebraicConstraintProjection` (for :math:`q = g(y)`
    algebraic constraints such as flux balance or constitutive laws) and
    a :class:`MuScaledSOCProjection` (for :math:`\mu`-scaled second-order
    cone contact) into a single projection that operates on disjoint
    index sets of the augmented state vector.

    The two sub-projections **must act on disjoint DOF indices**.
    Specifically:

    * The algebraic ``q_slice`` indices must not overlap any SOC block
      index.
    * The algebraic ``y_slice`` indices *may* overlap SOC block indices
      (e.g. if an algebraic constraint reads from a reaction DOF), but
      the *write* sets (``q_slice`` and SOC block indices) must be
      disjoint.

    The composite projection is:

    .. math::
        \Pi_{\text{comp}}(z) = \Pi_{\text{SOC}}\bigl(\Pi_{\text{alg}}(z)\bigr)

    Since the write sets are disjoint, the order does not matter
    for the projection value.  For the Clarke sub-differential
    (tangent cone), the composition rule simplifies to:

    .. math::
        D\Pi_{\text{comp}} = D\Pi_{\text{alg}} + D\Pi_{\text{SOC}} - I

    because each sub-projection is the identity on the other's write
    indices.

    Parameters
    ----------
    algebraic_projection : AlgebraicConstraintProjection
        Projection enforcing :math:`q = g(y)` algebraic constraints.
    soc_projection : MuScaledSOCProjection
        SOC cone projector for frictional contact reactions.
    component_slices : list of slice, optional
        Block partition forwarded to the base class.

    Notes
    -----
    This class does **not** set ``rho_independent = True``, so the
    nonlinear solver uses the semismooth Newton path — which correctly
    handles both the algebraic constraints (through their
    ``project`` / ``tangent_cone``) and the SOC nonlinearity (through
    the natural-map formulation with Clarke Jacobian).

    The ``lock_active_set``, ``unlock_active_set``, and
    ``reset_branch_cache`` methods are delegated to the SOC
    sub-projection for gap-based activation control.

    Examples
    --------
    >>> import numpy as np
    >>> from solve_nivp.projections import (
    ...     AlgebraicConstraintProjection,
    ...     MuScaledSOCProjection,
    ...     CompositeContactProjection,
    ... )
    >>> # Algebraic constraint: q = C @ y  on indices 2..3
    >>> C = np.array([[1.0, 0.5], [0.0, 1.0]])
    >>> alg = AlgebraicConstraintProjection(
    ...     g=lambda y: C @ y, dg_dy=lambda y: C,
    ...     y_slice=slice(0, 2), q_slice=slice(2, 4),
    ... )
    >>> # SOC contact on reaction DOFs 4..5
    >>> soc = MuScaledSOCProjection(
    ...     blocks=[(4, [5])],
    ...     get_mu=lambda y: 0.3,
    ...     zero_inactive=True,
    ... )
    >>> comp = CompositeContactProjection(alg, soc)
    >>> z = np.array([1., 2., 0., 0., 0.5, 0.1])
    >>> p = comp.project(z, z)
    >>> p[2:4]  # algebraic: C @ [1, 2] = [2, 2]
    array([2., 2.])
    """

    def __init__(self, algebraic_projection, soc_projection,
                 component_slices=None):
        super().__init__(component_slices=component_slices)

        if not isinstance(algebraic_projection, AlgebraicConstraintProjection):
            raise TypeError(
                "algebraic_projection must be an AlgebraicConstraintProjection "
                f"(got {type(algebraic_projection).__name__})")
        if not isinstance(soc_projection, MuScaledSOCProjection):
            raise TypeError(
                "soc_projection must be a MuScaledSOCProjection "
                f"(got {type(soc_projection).__name__})")

        self._alg = algebraic_projection
        self._soc = soc_projection

        # Validate disjoint write sets
        self._validate_disjoint_writes()

        # Cache identity matrix for tangent_cone combination
        self._eye_cache = {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_disjoint_writes(self):
        """Ensure algebraic q_slices and SOC block indices don't overlap."""
        # Collect all algebraic write indices
        alg_write = set()
        for blk in self._alg._blocks:
            # Use a generous upper bound for slice resolution
            ub = max((blk.q_slice.stop or 0), (blk.y_slice.stop or 0)) + 1
            alg_write.update(range(*blk.q_slice.indices(ub)))

        # Collect all SOC write indices
        soc_write = set()
        for idx_arr in self._soc._all_block_indices:
            soc_write.update(int(i) for i in idx_arr)

        overlap = alg_write & soc_write
        if overlap:
            raise ValueError(
                f"Algebraic q_slice indices and SOC block indices overlap "
                f"at {sorted(overlap)}.  The composite projection requires "
                f"disjoint write sets.")

    # ------------------------------------------------------------------
    # Projection API
    # ------------------------------------------------------------------
    def project(self, current_state, candidate, rhok=None, t=None,
                Fk_val=None, prev_state=None, step_size=None, **kw):
        """Apply algebraic constraints then SOC projection.

        Since the two sub-projections write to disjoint index sets,
        the order does not affect the result.
        """
        # Algebraic: q = g(y) on q_slices
        out = self._alg.project(current_state, candidate, rhok=rhok,
                                t=t, Fk_val=Fk_val)
        # SOC: project reaction block onto K_μ
        out = self._soc.project(current_state, out, rhok=rhok, t=t,
                                Fk_val=Fk_val, prev_state=prev_state,
                                step_size=step_size, **kw)
        return out

    def tangent_cone(self, candidate, current_state, rhok=None, t=None,
                     Fk_val=None, prev_state=None, step_size=None, **kw):
        r"""Clarke sub-differential of the composite projection.

        Since the two sub-projections act on disjoint write sets and
        each is the identity on the other's indices:

        .. math::
            D\Pi_{\text{comp}} = D\Pi_{\text{alg}} + D\Pi_{\text{SOC}} - I

        Returns
        -------
        D : scipy.sparse.csr_matrix or ndarray, shape (n, n)
        """
        D_alg = self._alg.tangent_cone(candidate, current_state,
                                        rhok=rhok, t=t, Fk_val=Fk_val)
        D_soc = self._soc.tangent_cone(candidate, current_state,
                                        rhok=rhok, t=t, Fk_val=Fk_val,
                                        prev_state=prev_state,
                                        step_size=step_size, **kw)
        n = candidate.shape[0]

        if sp.issparse(D_alg) or sp.issparse(D_soc):
            # Sparse path
            if not sp.issparse(D_alg):
                D_alg = sp.csr_matrix(D_alg)
            if not sp.issparse(D_soc):
                D_soc = sp.csr_matrix(D_soc)
            I = self._eye_cache.get(('csr', n))
            if I is None:
                I = sp.eye(n, format='csr')
                self._eye_cache[('csr', n)] = I
            return (D_alg + D_soc - I).tocsr()
        else:
            # Dense path (small systems)
            I = self._eye_cache.get(n)
            if I is None:
                I = np.eye(n)
                self._eye_cache[n] = I
            return np.asarray(D_alg) + np.asarray(D_soc) - I

    def project_batch(self, current_state, candidates, rhok=None,
                      t=None, Fk_val=None):
        """Batched projection: algebraic then SOC, row by row."""
        out = self._alg.project_batch(current_state, candidates,
                                       rhok=rhok, t=t, Fk_val=Fk_val)
        C = np.asarray(out, dtype=float)
        if C.ndim == 1:
            return self._soc.project(current_state, C, rhok=rhok,
                                     t=t, Fk_val=Fk_val)
        for i in range(C.shape[0]):
            C[i] = self._soc.project(current_state, C[i], rhok=rhok,
                                     t=t, Fk_val=Fk_val)
        return C

    # ------------------------------------------------------------------
    # Gap-based activation (delegated to SOC sub-projection)
    # ------------------------------------------------------------------
    def lock_active_set(self, y, t=None, reset_branch=True):
        """Delegate active-set locking to the SOC sub-projection."""
        self._soc.lock_active_set(y, t=t, reset_branch=reset_branch)

    def unlock_active_set(self):
        """Delegate active-set unlocking to the SOC sub-projection."""
        self._soc.unlock_active_set()

    def reset_branch_cache(self):
        """Delegate branch-cache reset to the SOC sub-projection."""
        self._soc.reset_branch_cache()

    # ------------------------------------------------------------------
    # Sub-projection accessors
    # ------------------------------------------------------------------
    @property
    def algebraic_projection(self):
        """The wrapped :class:`AlgebraicConstraintProjection`."""
        return self._alg

    @property
    def soc_projection(self):
        """The wrapped :class:`MuScaledSOCProjection`."""
        return self._soc

    @property
    def blocks(self):
        """SOC block list (forwarded from the SOC sub-projection)."""
        return self._soc.blocks

    @property
    def gap_func(self):
        """Gap function (forwarded from the SOC sub-projection)."""
        return self._soc.gap_func

    @property
    def gap_tol(self):
        """Gap tolerance (forwarded from the SOC sub-projection)."""
        return self._soc.gap_tol

    @property
    def zero_inactive(self):
        """Whether inactive SOC blocks are zeroed."""
        return self._soc.zero_inactive

    # Proxy _locked_active and _gap_nargs so the SSN solver's
    # Proposal-3 monotone-relocking can read/write them directly.
    @property
    def _locked_active(self):
        return self._soc._locked_active

    @_locked_active.setter
    def _locked_active(self, value):
        self._soc._locked_active = value

    @property
    def _gap_nargs(self):
        return self._soc._gap_nargs


##############################################################################
# MoreauSOCProjection — De Saxcé + restitution on top of geometric SOC
##############################################################################
class MoreauSOCProjection(MuScaledSOCProjection):
    r"""Moreau time-stepping with De Saxcé augmentation.

    Wraps forward / inverse De Saxcé transforms around the purely geometric
    SOC projection from :class:`MuScaledSOCProjection`.

    For each active block ``(s_idx, w_idx)`` the projection proceeds:

    1. **Forward De Saxcé** (on the candidate, which is ``y − ρ F(y)``):

       .. math::
           \tilde z_N = z_N + \alpha \|v_T\| + e\, v_N^{\mathrm{prev}},
           \qquad \tilde z_T = z_T

       where :math:`\alpha = \mu - \beta` (with :math:`\beta = \tan\psi`
       the dilatancy coefficient, default 0),
       :math:`v_T = y[\text{w\_idx}]` from the current iterate
       and :math:`v_N^{\mathrm{prev}} = y_{\text{prev}}[\text{s\_idx}]`.

    2. **Project** :math:`(\tilde z_N, \tilde z_T)` onto :math:`K_{1/\mu}`
       using the base-class cone projector.

    3. **Inverse De Saxcé** to recover the physical velocity:

       .. math::
           v_T = \tilde z_T^{\mathrm{proj}}, \qquad
           v_N = \tilde z_N^{\mathrm{proj}} - \alpha |\tilde z_T^{\mathrm{proj}}|
                 - e\, v_N^{\mathrm{prev}}

    This implements the complementarity

    .. math::
        \hat u \in K_{1/\mu},\; R \in K_\mu,\;
        \langle \hat u, R \rangle = 0

    which is exactly the Signorini + Coulomb contact law via De Saxcé's
    bipotential.

    Parameters
    ----------
    blocks : list of (int, array-like of int)
        Same as :class:`MuScaledSOCProjection`.  ``s_idx`` is the
        *normal-velocity* DOF, ``w_idx`` the tangential DOFs.
    get_mu : callable
        **Physical** friction coefficient (same signature as the base class).
    get_beta : callable or float or None, default None
        Dilatancy coefficient :math:`\beta = \tan\psi`.  Same calling
        convention as ``get_mu`` (scalar or per-block).  The De Saxcé
        augmentation uses :math:`\alpha = \mu - \beta` instead of
        :math:`\mu`.  Must satisfy :math:`0 \le \beta \le \mu` (enforced
        at evaluation time).  If *None* or ``0``, reduces to the
        standard non-dilatant formulation.
    gap_func : callable or None
        Same as :class:`MuScaledSOCProjection`.
    gap_tol : float
        Same as :class:`MuScaledSOCProjection`.
    e : float, default 0.0
        Coefficient of restitution.  ``e = 0`` is perfectly inelastic
        (no bounce); ``e = 1`` is perfectly elastic.
    component_slices : list of slice, optional
        Forwarded to the base class.

    Notes
    -----
    The forward De Saxcé adds terms that depend on ``current_state`` and
    ``prev_state``  (treated as constants in the tangent cone).  The
    resulting tangent-cone Jacobian accounts for the inverse De Saxcé
    chain rule but NOT for the dependence of :math:`v_T` on the candidate
    — this is consistent with the operator-splitting philosophy of the
    Uzawa / proximal-point iteration used in the VI solver.
    """

    def __init__(self, *, blocks, get_mu, get_beta=None, gap_func=None,
                 gap_tol=0.0, e=0.0, zero_inactive=False,
                 get_s0=None, get_w0=None,
                 get_ds0_dz=None, get_dw0_dz=None,
                 component_slices=None):
        self._e = float(e)
        super().__init__(
            blocks=blocks,
            get_mu=get_mu,          # physical μ
            gap_func=gap_func,
            gap_tol=gap_tol,
            zero_inactive=zero_inactive,
            get_s0=get_s0,
            get_w0=get_w0,
            get_ds0_dz=get_ds0_dz,
            get_dw0_dz=get_dw0_dz,
            component_slices=component_slices,
        )
        # Dilatancy: get_beta callback (same convention as get_mu)
        if get_beta is None:
            self.get_beta = None
            self._beta_nargs = None
        elif callable(get_beta):
            self.get_beta = get_beta
            self._beta_nargs = _count_required_args(get_beta)
        else:
            # Scalar constant
            _b = float(get_beta)
            self.get_beta = lambda y, _b=_b: _b
            self._beta_nargs = 1

    # ------------------------------------------------------------------
    # Dilatancy evaluation
    # ------------------------------------------------------------------
    def _eval_beta(self, y, mu_arr, t=None, Fk_val=None):
        """Evaluate dilatancy and return ``alpha = mu - beta`` per block.

        Returns ``mu_arr`` unchanged when ``get_beta`` is None.
        """
        if self.get_beta is None:
            return mu_arr.copy()

        nargs = self._beta_nargs
        if nargs is None or nargs >= 3:
            try:
                beta_val = self.get_beta(y, t, Fk_val)
            except TypeError:
                beta_val = (self.get_beta(y, t) if nargs != 1
                            else self.get_beta(y))
        elif nargs == 2:
            beta_val = self.get_beta(y, t)
        else:
            beta_val = self.get_beta(y)

        beta_arr = np.atleast_1d(np.asarray(beta_val, dtype=float))
        nb = len(self.blocks)
        if beta_arr.size == 1:
            beta_arr = np.full(nb, float(beta_arr.flat[0]))
        elif beta_arr.size == nb:
            beta_arr = beta_arr.ravel()
        else:
            raise ValueError(
                f"get_beta must return a scalar or array of length {nb} "
                f"(got size {beta_arr.size})")
        # Validate: 0 <= beta <= mu
        if not np.all(np.isfinite(beta_arr)):
            raise ValueError(
                f"get_beta returned non-finite value(s): {beta_arr}")
        if np.any(beta_arr < -1e-14):
            raise ValueError(
                f"get_beta returned negative value(s): {beta_arr}. "
                f"Dilatancy coefficient must be >= 0.")
        if np.any(beta_arr > mu_arr + 1e-14):
            raise ValueError(
                f"get_beta returned value(s) exceeding mu: "
                f"beta={beta_arr}, mu={mu_arr}. "
                f"Must have beta <= mu.")
        return mu_arr - np.clip(beta_arr, 0.0, mu_arr)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(self, current_state, candidate, rhok=None, t=None,
                Fk_val=None, prev_state=None, step_size=None, **kw):
        y = np.asarray(current_state, dtype=float)
        z = np.asarray(candidate, dtype=float).copy()

        active = self._active_mask(y, t)
        mu_phys = self._eval_mu(y, t=t, Fk_val=Fk_val)
        alpha_arr = self._eval_beta(y, mu_phys, t=t, Fk_val=Fk_val)

        # Pre-stress evaluation (skip when no callbacks configured)
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            mu_k = float(mu_phys[k])
            alpha_k = float(alpha_arr[k])   # mu - beta (De Saxcé coeff)
            idx = self._all_block_indices[k]

            # Current tangential velocity from iterate (constant in Jacobian)
            v_T = y[w_idx]
            v_T_norm = float(np.linalg.norm(v_T))

            # Previous normal velocity for restitution
            v_N_prev = float(prev_state[s_idx]) if prev_state is not None else 0.0

            # ---- Forward De Saxcé (uses alpha = mu - beta) ----
            z[s_idx] += alpha_k * v_T_norm + self._e * v_N_prev
            # z_T unchanged — already contains v_T - rho*R_T from VI candidate

            # ---- Project onto K_{1/mu} (exact, no persistence) ----
            # The yield cone is still K_{1/mu} — only augmentation uses alpha
            z_blk = z[idx]
            # Apply pre-stress shift (translate cone)
            if _has_ps:
                z_blk = z_blk.copy()
                s0_k = float(s0_arr[k])
                w0_k = self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                z_blk[0] += s0_k
                z_blk[1:] += w0_k
            if mu_k > 0.0:
                z_proj = self._proj_mu_scaled_soc(z_blk, 1.0 / mu_k)
            else:
                # mu=0: frictionless — K_{inf} = {s >= 0, any w}
                z_proj = z_blk.copy()
                z_proj[0] = max(0.0, z_proj[0])
            # Un-shift pre-stress
            if _has_ps:
                z_proj[0] -= s0_k
                z_proj[1:] -= w0_k

            # ---- Inverse De Saxcé (uses alpha = mu - beta) ----
            v_T_proj_norm = float(np.linalg.norm(z_proj[1:]))
            z_proj[0] -= alpha_k * v_T_proj_norm + self._e * v_N_prev

            z[idx] = z_proj

        return z

    # ------------------------------------------------------------------
    # Tangent cone (Clarke sub-differential with De Saxcé chain rule)
    # ------------------------------------------------------------------
    def tangent_cone(self, candidate, current_state, rhok=None, t=None,
                     Fk_val=None, prev_state=None, step_size=None, **kw):
        r"""Tangent cone with composed De Saxcé Jacobian.

        Chain rule: :math:`J = J_{\text{inv}} \cdot J_{\Pi_{K_{1/\mu}}}`
        (the forward De Saxcé Jacobian w.r.t. *z* is the identity since
        the added terms depend on ``current_state``, not *z*).

        When pre-stress derivatives are provided, the correction
        :math:`J_{\text{inv}} (J_\pi - I)\,\partial z_0/\partial z`
        is added.
        """
        y = np.asarray(current_state, dtype=float)
        z = np.asarray(candidate, dtype=float).copy()
        n = z.size

        active = self._active_mask(y, t)
        mu_phys = self._eval_mu(y, t=t, Fk_val=Fk_val)
        alpha_arr = self._eval_beta(y, mu_phys, t=t, Fk_val=Fk_val)
        nz_tol = 1e-15

        # Apply forward De Saxcé so we evaluate the cone Jacobian at the
        # correct (transformed) point.  Uses alpha = mu - beta.
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            alpha_k = float(alpha_arr[k])
            v_T_norm = float(np.linalg.norm(y[w_idx]))
            v_N_prev = float(prev_state[s_idx]) if prev_state is not None else 0.0
            z[s_idx] += alpha_k * v_T_norm + self._e * v_N_prev

        # Pre-compute block Jacobians ONCE per active block (avoids
        # redundant calls to _proj_mu_scaled_soc).
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        # Pre-stress Jacobian evaluation (state-dependent offset)
        _has_ps_jac = self.get_ds0_dz is not None or self.get_dw0_dz is not None
        ds0_dz_all = (self._eval_ds0_dz(y, t=t, Fk_val=Fk_val)
                      if _has_ps_jac and self.get_ds0_dz is not None
                      else None)

        block_data = {}  # k -> (J_composed, ps_corr or None)
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            mu_k = float(mu_phys[k])
            alpha_k = float(alpha_arr[k])   # mu - beta
            idx = self._all_block_indices[k]
            m = idx.size - 1  # tangential dimension
            z_blk = z[idx]

            # Apply pre-stress shift (translate cone)
            if _has_ps:
                z_blk = z_blk.copy()
                s0_k = float(s0_arr[k])
                w0_k = self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                z_blk[0] += s0_k
                z_blk[1:] += w0_k

            if mu_k > 0.0:
                z_proj_val, J_pi = self._proj_persistent(
                    z_blk, 1.0 / mu_k, block_key=('moreau', k),
                    return_jacobian=True)
            else:
                d = idx.size
                J_pi = np.eye(d)
                z_proj_val = z_blk.copy()
                if z_blk[0] < 0.0:
                    J_pi[0, :] = 0.0
                z_proj_val[0] = max(0.0, z_proj_val[0])

            # Un-shift pre-stress from projected value for inverse De Saxcé
            if _has_ps:
                z_proj_val = z_proj_val.copy()
                z_proj_val[0] -= s0_k
                z_proj_val[1:] -= w0_k

            # Inverse De Saxcé Jacobian (uses alpha = mu - beta)
            w_proj = z_proj_val[1:]
            w_proj_norm = float(np.linalg.norm(w_proj))
            J_inv = np.eye(1 + m)
            if alpha_k > 0.0 and w_proj_norm > nz_tol:
                w_hat = w_proj / w_proj_norm
                J_inv[0, 1:] = -alpha_k * w_hat

            J_composed = J_inv @ J_pi

            # State-dependent pre-stress correction:
            # J_inv @ (J_pi - I) @ dz0_k/dz
            ps_corr = None
            if _has_ps_jac:
                dz0_k = self._assemble_dz0_block(
                    y, k, t, Fk_val, n, ds0_dz_all)
                if dz0_k is not None:
                    d_k = J_pi.shape[0]
                    ps_corr = J_inv @ (J_pi - np.eye(d_k)) @ dz0_k

            block_data[k] = (J_composed, ps_corr)

        # --- Dense fast-path for small systems ---
        if n <= 64:
            D = np.eye(n)
            for k, (s_idx, w_idx) in enumerate(self.blocks):
                if k not in block_data:
                    continue
                idx = self._all_block_indices[k]
                J_composed, ps_corr = block_data[k]
                D[np.ix_(idx, idx)] = J_composed
                if ps_corr is not None:
                    D[idx, :] += ps_corr
            return D

        # --- Sparse path for larger systems ---
        block_row_map = {}
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            idx = self._all_block_indices[k]
            for local_r, global_r in enumerate(idx):
                block_row_map[int(global_r)] = (k, local_r, idx)

        data = []
        col_indices = []
        indptr = [0]

        for r in range(n):
            entry = block_row_map.get(r)
            if entry is None:
                data.append(1.0)
                col_indices.append(r)
                indptr.append(len(data))
                continue

            k, local_r, idx = entry
            J_composed, ps_corr = block_data[k]
            if ps_corr is not None:
                full_row = np.zeros(n)
                full_row[idx] = J_composed[local_r, :]
                full_row += ps_corr[local_r, :]
                nz = np.flatnonzero(np.abs(full_row) > nz_tol)
                if nz.size:
                    data.extend(full_row[nz].tolist())
                    col_indices.extend(nz.tolist())
            else:
                row_vals = J_composed[local_r, :]
                nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
                if nz.size:
                    data.extend(row_vals[nz].tolist())
                    col_indices.extend(idx[nz].tolist())
            indptr.append(len(data))

        return sp.csr_matrix(
            (np.array(data, dtype=float),
             np.array(col_indices, dtype=int),
             np.array(indptr, dtype=int)),
            shape=(n, n))


    # ------------------------------------------------------------------
    # Tangent cone split: derivatives wrt candidate and current_state
    # ------------------------------------------------------------------
    def tangent_cone_split(self, candidate, current_state, rhok=None, t=None,
                           Fk_val=None, prev_state=None, step_size=None, **kw):
        r"""Return a pair ``(D_cand, D_state)``.

        * ``D_cand`` is the Clarke selection of ``∂P/∂candidate``.
        * ``D_state`` is the generalized derivative of ``P`` with respect
          to ``current_state`` (holding ``candidate`` fixed).

        This is needed for the full semismooth Newton Jacobian of the
        natural residual ``r(y) = y - P(y, y - lam F(y))`` when ``P`` is
        state-dependent (De Saxcé augmentation, state-dependent pre-stress).

        The dominant state-dependence for Moreau contact is the De Saxcé
        normal augmentation ``+ alpha ||v_T||`` where ``alpha = mu - beta``.
        This method includes its nonsmooth derivative using a minimal-norm
        subgradient at ``||v_T|| = 0`` (zero vector).
        """
        y = np.asarray(current_state, dtype=float)
        z = np.asarray(candidate, dtype=float).copy()
        n = z.size

        active = self._active_mask(y, t)
        mu_phys = self._eval_mu(y, t=t, Fk_val=Fk_val)
        alpha_arr = self._eval_beta(y, mu_phys, t=t, Fk_val=Fk_val)
        nz_tol = 1e-15

        # Apply forward De Saxcé to the candidate so J_pi is evaluated at
        # the correct transformed point.  Uses alpha = mu - beta.
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            alpha_k = float(alpha_arr[k])
            v_T_norm = float(np.linalg.norm(y[w_idx]))
            v_N_prev = float(prev_state[s_idx]) if prev_state is not None else 0.0
            z[s_idx] += alpha_k * v_T_norm + self._e * v_N_prev

        # Pre-stress evaluation (skip when no callbacks configured)
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        # Pre-stress Jacobian evaluation (state-dependent offset)
        _has_ps_jac = self.get_ds0_dz is not None or self.get_dw0_dz is not None
        ds0_dz_all = (self._eval_ds0_dz(y, t=t, Fk_val=Fk_val)
                      if _has_ps_jac and self.get_ds0_dz is not None
                      else None)

        # Per-block data:
        #   A_blk: dxd   (wrt candidate)
        #   B_blk: dxd   (state dependence local to block, De Saxcé term)
        #   B_ps : dxn   (state-dependent pre-stress correction, possibly global)
        block_data = {}  # k -> (A_blk, B_blk, B_ps or None)
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            mu_k = float(mu_phys[k])
            alpha_k = float(alpha_arr[k])   # mu - beta
            idx = self._all_block_indices[k]
            m = idx.size - 1
            z_blk = z[idx]

            # Apply pre-stress shift for evaluation of cone Jacobian
            if _has_ps:
                z_blk = z_blk.copy()
                s0_k = float(s0_arr[k])
                w0_k = self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                z_blk[0] += s0_k
                z_blk[1:] += w0_k

            if mu_k > 0.0:
                z_proj_val, J_pi = self._proj_persistent(
                    z_blk, 1.0 / mu_k, block_key=('moreau', k),
                    return_jacobian=True)
            else:
                d = idx.size
                J_pi = np.eye(d)
                z_proj_val = z_blk.copy()
                if z_blk[0] < 0.0:
                    J_pi[0, :] = 0.0
                z_proj_val[0] = max(0.0, z_proj_val[0])

            # Un-shift for inverse De Saxcé and for J_inv evaluation
            if _has_ps:
                z_proj_val = z_proj_val.copy()
                z_proj_val[0] -= s0_k
                z_proj_val[1:] -= w0_k

            # Inverse De Saxcé Jacobian (uses alpha = mu - beta)
            w_proj = z_proj_val[1:]
            w_proj_norm = float(np.linalg.norm(w_proj))
            J_inv = np.eye(1 + m)
            if alpha_k > 0.0 and w_proj_norm > nz_tol:
                w_hat = w_proj / w_proj_norm
                J_inv[0, 1:] = -alpha_k * w_hat

            A_blk = J_inv @ J_pi

            # --- State dependence from forward De Saxcé:  z_N += alpha ||v_T(y)||
            # The forward De Saxcé adds a term that depends on current_state
            # (not candidate).  Its derivative w.r.t. current_state is:
            #   dP/dy|_fwd = J_inv @ J_pi @ Fwd
            # where Fwd[0, w_idx_local] = alpha * v_T / ||v_T|| (the subgradient
            # of the norm at v_T), and Fwd is zero elsewhere.
            B_blk = np.zeros((1 + m, 1 + m), dtype=float)
            if alpha_k > 0.0:
                v_T = y[w_idx]
                v_T_norm = float(np.linalg.norm(v_T))
                if v_T_norm > nz_tol:
                    grad = (alpha_k / v_T_norm) * np.asarray(v_T, dtype=float)  # length m
                    Fwd = np.zeros((1 + m, 1 + m), dtype=float)
                    Fwd[0, 1:] = grad
                    B_blk = A_blk @ Fwd
                # else: minimal-norm Clarke selection = 0

            # --- State-dependent pre-stress correction belongs to D_state
            B_ps = None
            if _has_ps_jac:
                dz0_k = self._assemble_dz0_block(
                    y, k, t, Fk_val, n, ds0_dz_all)
                if dz0_k is not None:
                    d_k = J_pi.shape[0]
                    B_ps = J_inv @ (J_pi - np.eye(d_k)) @ dz0_k

            block_data[k] = (A_blk, B_blk, B_ps)

        # --- Dense path for small systems ---
        if n <= 64:
            D_cand = np.eye(n)
            D_state = np.zeros((n, n))
            for k in block_data:
                idx = self._all_block_indices[k]
                A_blk, B_blk, B_ps = block_data[k]
                D_cand[np.ix_(idx, idx)] = A_blk
                D_state[np.ix_(idx, idx)] += B_blk
                if B_ps is not None:
                    D_state[idx, :] += B_ps
            return D_cand, D_state

        # --- Sparse assembly ---
        block_row_map = {}
        for k in block_data:
            idx = self._all_block_indices[k]
            for local_r, global_r in enumerate(idx):
                block_row_map[int(global_r)] = (k, local_r, idx)

        # Assemble D_cand (identity with block-row replacements)
        data_A = []
        col_A = []
        indptr_A = [0]

        # Assemble D_state (mostly zero with sparse block rows)
        data_B = []
        col_B = []
        indptr_B = [0]

        for r in range(n):
            entry = block_row_map.get(r)

            # ---- D_cand row ----
            if entry is None:
                data_A.append(1.0)
                col_A.append(r)
            else:
                k, local_r, idx = entry
                A_blk, B_blk, B_ps = block_data[k]
                row_vals = A_blk[local_r, :]
                nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
                if nz.size:
                    data_A.extend(row_vals[nz].tolist())
                    col_A.extend(idx[nz].tolist())
            indptr_A.append(len(data_A))

            # ---- D_state row ----
            if entry is None:
                # empty row
                indptr_B.append(len(data_B))
                continue

            k, local_r, idx = entry
            A_blk, B_blk, B_ps = block_data[k]

            if B_ps is not None:
                full_row = np.zeros(n)
                # local block contribution
                full_row[idx] += B_blk[local_r, :]
                # global pre-stress contribution
                full_row += B_ps[local_r, :]
                nz = np.flatnonzero(np.abs(full_row) > nz_tol)
                if nz.size:
                    data_B.extend(full_row[nz].tolist())
                    col_B.extend(nz.tolist())
            else:
                row_vals = B_blk[local_r, :]
                nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
                if nz.size:
                    data_B.extend(row_vals[nz].tolist())
                    col_B.extend(idx[nz].tolist())

            indptr_B.append(len(data_B))

        D_cand = sp.csr_matrix(
            (np.array(data_A, dtype=float),
             np.array(col_A, dtype=int),
             np.array(indptr_A, dtype=int)),
            shape=(n, n))

        D_state = sp.csr_matrix(
            (np.array(data_B, dtype=float),
             np.array(col_B, dtype=int),
             np.array(indptr_B, dtype=int)),
            shape=(n, n))

        return D_cand, D_state

##############################################################################
# AnisotropicSOCProjection — elliptic friction cone projector
##############################################################################
class AnisotropicSOCProjection(MuScaledSOCProjection):
    r"""Anisotropic friction cone projector via Cholesky whitening.

    For each block the constraint is

    .. math::
        (s, \mathbf{w}) \in K_{\mu,B}
        = \bigl\{(s,\mathbf{w}) : s \ge 0,\;
          \sqrt{\mathbf{w}^T B\,\mathbf{w}} \le \mu\, s \bigr\}

    where :math:`B \succ 0` is a symmetric positive-definite matrix in the
    tangential subspace.  The isotropic case corresponds to
    :math:`B = I`.

    **Implementation**: the elliptic cone is reduced to the standard
    isotropic cone by a Cholesky whitening transform.  Let
    :math:`B = L L^T` (Cholesky).  Define :math:`\mathbf{v} = L^T\mathbf{w}`.
    Then

    .. math::
        \|\mathbf{w}\|_B = \|L^T\mathbf{w}\| = \|\mathbf{v}\|

    so the constraint becomes the standard :math:`\|\mathbf{v}\| \le \mu s`.
    We project :math:`\tilde z = (s, L^T \mathbf{w})` onto
    :math:`K_\mu` using the spectral isotropic projector, then transform
    back: :math:`\mathbf{w}_{\mathrm{proj}} = L^{-T}\, \mathbf{v}_{\mathrm{proj}}`.

    The Jacobian follows by chain rule:

    .. math::
        J = T^{-1}\, J_{\mathrm{iso}}(\tilde z)\, T,
        \quad T = \begin{pmatrix} 1 & 0 \\ 0 & L^T \end{pmatrix}

    This approach:

    * Is exact (no iterative root-find).
    * Handles all edge cases (apex, :math:`s \le 0`, etc.)
      because it delegates to the battle-tested spectral projector.
    * Recovers the isotropic projector when :math:`B = I`.

    Parameters
    ----------
    blocks : list of (int, array-like of int)
        Same as :class:`MuScaledSOCProjection`.
    get_mu : callable
        Friction coefficient callback (same as base class).
    get_B : callable
        Returns the anisotropy matrix per block.  Signature
        ``get_B(y, k) -> ndarray(m, m)`` where *k* is the block index
        and *m* is the tangential dimension of that block.  Must return
        a symmetric positive-definite matrix.  For isotropic friction
        return ``np.eye(m)``.
    gap_func, gap_tol, component_slices
        Forwarded to the base class.

    Notes
    -----
    Different friction coefficients per tangential direction can be
    encoded as :math:`B = \mathrm{diag}(1/\mu_1^2, 1/\mu_2^2)` with
    :math:`\mu = 1`.  Direction-dependent friction aligned with a local
    basis is obtained by rotating :math:`B`.
    """

    def __init__(self, *, blocks, get_mu, get_B, gap_func=None,
                 gap_tol=0.0, zero_inactive=False,
                 get_s0=None, get_w0=None,
                 get_ds0_dz=None, get_dw0_dz=None,
                 component_slices=None):
        super().__init__(
            blocks=blocks,
            get_mu=get_mu,
            gap_func=gap_func,
            gap_tol=gap_tol,
            zero_inactive=zero_inactive,
            get_s0=get_s0,
            get_w0=get_w0,
            get_ds0_dz=get_ds0_dz,
            get_dw0_dz=get_dw0_dz,
            component_slices=component_slices,
        )
        self.get_B = get_B
        # Cache Cholesky factors per block
        self._chol_cache = {}    # block_key -> (L, L_inv_T)

    # ------------------------------------------------------------------
    # Cholesky evaluation and caching
    # ------------------------------------------------------------------
    def _eval_chol(self, y, k):
        r"""Evaluate and cache Cholesky factors for block *k*.

        Returns ``(L, L_inv_T)`` where ``B = L @ L.T`` and
        ``L_inv_T = inv(L).T = inv(L.T)``.
        """
        B = np.asarray(self.get_B(y, k), dtype=float)
        L = np.linalg.cholesky(B)             # B = L L^T
        L_inv_T = np.linalg.inv(L).T          # L^{-T}
        self._chol_cache[k] = (L, L_inv_T)
        return L, L_inv_T

    # ------------------------------------------------------------------
    # Core anisotropic projector — Cholesky whitening + isotropic
    # ------------------------------------------------------------------
    @staticmethod
    def _proj_anisotropic(z, mu, B, B_inv, return_jacobian=False,
                          **_ignored):
        r"""Project ``z = (s, w)`` onto :math:`K_{\mu, B}`.

        Uses Cholesky whitening: :math:`B = L L^T`,
        :math:`\tilde z = (s, L^T w)`, project isotropically, then
        :math:`w_{\mathrm{proj}} = L^{-T} v_{\mathrm{proj}}`.

        Parameters
        ----------
        z : ndarray, shape (1+m,)
        mu : float
        B : ndarray, shape (m, m)
            SPD anisotropy matrix.
        B_inv : ndarray, shape (m, m)
            Precomputed inverse of B (used only for API compat;
            Cholesky is computed internally).
        return_jacobian : bool

        Returns
        -------
        p : ndarray, shape (1+m,)
            or ``(p, J)`` when ``return_jacobian=True``.
        """
        z = np.asarray(z, dtype=float)
        s = float(z[0])
        w = z[1:].copy()
        m = w.size
        d = 1 + m

        # Cholesky: B = L L^T
        L = np.linalg.cholesky(B)
        LT = L.T
        L_inv_T = np.linalg.inv(LT)    # L^{-T}

        # Transform: v = L^T w
        v = LT @ w

        # Build transformed vector z̃ = (s, v)
        z_tilde = np.empty(d)
        z_tilde[0] = s
        z_tilde[1:] = v

        # Isotropic projection of z̃ onto K_μ
        if return_jacobian:
            p_tilde, J_iso = MuScaledSOCProjection._proj_mu_scaled_soc(
                z_tilde, mu, return_jacobian=True)
        else:
            p_tilde = MuScaledSOCProjection._proj_mu_scaled_soc(
                z_tilde, mu, return_jacobian=False)

        # Inverse transform: w_proj = L^{-T} v_proj
        p = np.empty(d)
        p[0] = p_tilde[0]
        p[1:] = L_inv_T @ p_tilde[1:]

        if not return_jacobian:
            return p

        # Jacobian via chain rule: J = T^{-1} J_iso T
        # where T = diag(1, L^T), T^{-1} = diag(1, L^{-T})
        #
        # T^{-1} J_iso T in block form:
        #   J[0,0] = J_iso[0,0]
        #   J[0,1:] = J_iso[0,1:] @ L^T
        #   J[1:,0] = L^{-T} @ J_iso[1:,0]
        #   J[1:,1:] = L^{-T} @ J_iso[1:,1:] @ L^T
        J = np.empty((d, d))
        J[0, 0] = J_iso[0, 0]
        J[0, 1:] = J_iso[0, 1:] @ LT
        J[1:, 0] = L_inv_T @ J_iso[1:, 0]
        J[1:, 1:] = L_inv_T @ J_iso[1:, 1:] @ LT

        return p, J

    # ------------------------------------------------------------------
    # Override _proj_persistent to use anisotropic projector
    # ------------------------------------------------------------------
    def _proj_persistent(self, z, mu, block_key, return_jacobian=False,
                         *, L=None, L_inv_T=None):
        r"""Anisotropic projection with branch persistence.

        Transforms to isotropic coordinates via Cholesky whitening,
        delegates to the base-class spectral projector, and transforms
        back.

        Parameters
        ----------
        z, mu, block_key, return_jacobian
            Same as base class.
        L : ndarray or None
            Lower Cholesky factor of B.
        L_inv_T : ndarray or None
            Inverse transpose of L (i.e., L^{-T}).
        """
        if L is None:
            # No anisotropy info → isotropic base-class path
            return super()._proj_persistent(z, mu, block_key,
                                            return_jacobian=return_jacobian)

        z = np.asarray(z, dtype=float)
        w = z[1:]
        m = w.size
        d = 1 + m
        LT = L.T

        # Transform to isotropic coordinates
        z_tilde = np.empty(d)
        z_tilde[0] = z[0]
        z_tilde[1:] = LT @ w

        # Use base-class spectral projector (with persistence)
        result = super()._proj_persistent(z_tilde, mu, block_key,
                                          return_jacobian=return_jacobian)

        if return_jacobian:
            p_tilde, J_iso = result
            # Inverse transform
            p = np.empty(d)
            p[0] = p_tilde[0]
            p[1:] = L_inv_T @ p_tilde[1:]
            # Chain rule: J = T^{-1} J_iso T
            J = np.empty((d, d))
            J[0, 0] = J_iso[0, 0]
            J[0, 1:] = J_iso[0, 1:] @ LT
            J[1:, 0] = L_inv_T @ J_iso[1:, 0]
            J[1:, 1:] = L_inv_T @ J_iso[1:, 1:] @ LT
            return p, J
        else:
            p_tilde = result
            p = np.empty(d)
            p[0] = p_tilde[0]
            p[1:] = L_inv_T @ p_tilde[1:]
            return p

    # ------------------------------------------------------------------
    # Override project to pass Cholesky factors per block
    # ------------------------------------------------------------------
    def project(self, current_state, candidate, rhok=None, t=None,
                Fk_val=None, prev_state=None, step_size=None, **kw):
        y = np.asarray(current_state, dtype=float)
        z_work = np.asarray(candidate, dtype=float).copy()

        active = self._active_mask(y, t)
        mu_arr = self._eval_mu(y, t=t, Fk_val=Fk_val)

        # Pre-stress evaluation (skip when no callbacks configured)
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        for k, (s_idx, w_idx) in enumerate(self.blocks):
            idx = self._all_block_indices[k]
            if not active[k]:
                if self.zero_inactive:
                    z_work[idx] = 0.0
                continue
            mu_k = float(mu_arr[k])
            z_blk = z_work[idx]
            # Apply pre-stress shift (translate cone)
            if _has_ps:
                z_blk = z_blk.copy()
                s0_k = float(s0_arr[k])
                w0_k = self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                z_blk[0] += s0_k
                z_blk[1:] += w0_k
            L, L_inv_T = self._eval_chol(y, k)
            # Exact projection (no persistence) via Cholesky whitening
            LT = L.T
            z_tilde = np.empty_like(z_blk)
            z_tilde[0] = z_blk[0]
            z_tilde[1:] = LT @ z_blk[1:]
            p_tilde = self._proj_mu_scaled_soc(
                z_tilde, mu_k, return_jacobian=False)
            z_proj = np.empty_like(z_blk)
            z_proj[0] = p_tilde[0]
            z_proj[1:] = L_inv_T @ p_tilde[1:]
            # Un-shift pre-stress
            if _has_ps:
                z_proj[0] -= s0_k
                z_proj[1:] -= w0_k
            z_work[idx] = z_proj

        return z_work

    # ------------------------------------------------------------------
    # Override tangent_cone to pass Cholesky factors per block
    # ------------------------------------------------------------------
    def tangent_cone(self, candidate, current_state, rhok=None, t=None,
                     Fk_val=None, prev_state=None, step_size=None, **kw):
        y = np.asarray(current_state, dtype=float)
        z = np.asarray(candidate, dtype=float)
        n = z.size

        active = self._active_mask(y, t)
        mu_arr = self._eval_mu(y, t=t, Fk_val=Fk_val)

        # Pre-stress evaluation (skip when no callbacks configured)
        _has_ps = self.get_s0 is not None or self.get_w0 is not None
        s0_arr = self._eval_s0(y, t=t, Fk_val=Fk_val) if _has_ps else None

        # Pre-stress Jacobian evaluation (state-dependent offset)
        _has_ps_jac = self.get_ds0_dz is not None or self.get_dw0_dz is not None
        ds0_dz_all = (self._eval_ds0_dz(y, t=t, Fk_val=Fk_val)
                      if _has_ps_jac and self.get_ds0_dz is not None
                      else None)

        if n <= 64:
            D = np.eye(n)
            for k, (s_idx, w_idx) in enumerate(self.blocks):
                idx = self._all_block_indices[k]
                if not active[k]:
                    if self.zero_inactive:
                        D[np.ix_(idx, idx)] = 0.0
                    continue
                mu_k = float(mu_arr[k])
                z_blk = z[idx]
                # Shift to pre-stressed coordinates for Jacobian evaluation
                if _has_ps:
                    z_blk = z_blk.copy()
                    z_blk[0] += float(s0_arr[k])
                    z_blk[1:] += self._eval_w0(y, k, t=t, Fk_val=Fk_val)
                L, L_inv_T = self._eval_chol(y, k)
                _, J_blk = self._proj_persistent(
                    z_blk, mu_k, block_key=k,
                    return_jacobian=True, L=L, L_inv_T=L_inv_T)
                D[np.ix_(idx, idx)] = J_blk
                # State-dependent pre-stress Jacobian correction:
                # D[idx, :] += (J_aniso - I) @ dz0_k/dz
                if _has_ps_jac:
                    dz0_k = self._assemble_dz0_block(
                        y, k, t, Fk_val, n, ds0_dz_all)
                    if dz0_k is not None:
                        d_k = J_blk.shape[0]
                        D[idx, :] += (J_blk - np.eye(d_k)) @ dz0_k
            return D

        block_jacs = {}  # k -> (J_blk, dz0_k or None)
        zero_block_indices = set()  # global indices zeroed (inactive)
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            idx = self._all_block_indices[k]
            if not active[k]:
                if self.zero_inactive:
                    zero_block_indices.update(int(i) for i in idx)
                continue
            mu_k = float(mu_arr[k])
            z_blk = z[idx]
            # Shift to pre-stressed coordinates for Jacobian evaluation
            if _has_ps:
                z_blk = z_blk.copy()
                z_blk[0] += float(s0_arr[k])
                z_blk[1:] += self._eval_w0(y, k, t=t, Fk_val=Fk_val)
            L, L_inv_T = self._eval_chol(y, k)
            _, J_blk = self._proj_persistent(
                z_blk, mu_k, block_key=k,
                return_jacobian=True, L=L, L_inv_T=L_inv_T)
            dz0_k = (self._assemble_dz0_block(y, k, t, Fk_val, n, ds0_dz_all)
                     if _has_ps_jac else None)
            block_jacs[k] = (J_blk, dz0_k)

        block_row_map = {}
        for k, (s_idx, w_idx) in enumerate(self.blocks):
            if not active[k]:
                continue
            idx = self._all_block_indices[k]
            for local_r, global_r in enumerate(idx):
                block_row_map[int(global_r)] = (k, local_r, idx)

        nz_tol = 1e-15
        data = []
        col_indices = []
        indptr = [0]

        for r in range(n):
            if r in zero_block_indices:
                # Inactive block with zero_inactive: zero row
                indptr.append(len(data))
                continue
            entry = block_row_map.get(r)
            if entry is None:
                data.append(1.0)
                col_indices.append(r)
                indptr.append(len(data))
                continue
            k, local_r, idx = entry
            J_blk, dz0_k = block_jacs[k]
            if dz0_k is not None:
                d_k = J_blk.shape[0]
                corr = ((J_blk - np.eye(d_k)) @ dz0_k)[local_r, :]
                full_row = np.zeros(n)
                full_row[idx] = J_blk[local_r, :]
                full_row += corr
                nz = np.flatnonzero(np.abs(full_row) > nz_tol)
                if nz.size:
                    data.extend(full_row[nz].tolist())
                    col_indices.extend(nz.tolist())
            else:
                row_vals = J_blk[local_r, :]
                nz = np.flatnonzero(np.abs(row_vals) > nz_tol)
                if nz.size:
                    data.extend(row_vals[nz].tolist())
                    col_indices.extend(idx[nz].tolist())
            indptr.append(len(data))

        return sp.csr_matrix(
            (np.array(data, dtype=float),
             np.array(col_indices, dtype=int),
             np.array(indptr, dtype=int)),
            shape=(n, n))



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
