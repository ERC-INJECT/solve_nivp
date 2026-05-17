#!/usr/bin/env python
"""Workspace-local CG extension that assembles traction boundary loads.

The upstream ``CGPoroelastostatics`` class normalizes ``bc['t']`` but does not
currently add the corresponding Neumann traction term to ``R0``.  This local
subclass fills that gap so notebooks in this workspace can run true
traction-driven benchmarks without patching the external package checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from skfem import FacetBasis, LinearForm

PORO_ROOT = Path("/home/david/Documents/Poroelasticity")
if PORO_ROOT.exists() and str(PORO_ROOT) not in sys.path:
    sys.path.insert(0, str(PORO_ROOT))

from poroelasticity.cgporoelastostatics import CGPoroelastostatics  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "4")


class TractionAwareCGPoroelastostatics(CGPoroelastostatics):
    """Add constant / callable traction BC assembly to the CG poro class."""

    def __init__(self, *args, traction_scale_fn=None, **kwargs):
        self._traction_scale_fn = traction_scale_fn
        self._R_traction = None
        self._traction_scale_t0 = 1.0
        super().__init__(*args, **kwargs)

    def _assemble_system(self):
        super()._assemble_system()
        self._assemble_traction_bc_rhs()

    def _evaluate_traction_scale(self, t_dim):
        """Return the scalar traction load factor at dimensional time ``t_dim``."""
        if self._traction_scale_fn is None:
            return 1.0
        scale = self._traction_scale_fn(float(t_dim))
        arr = np.asarray(scale, dtype=float).ravel()
        if arr.size != 1 or not np.isfinite(arr[0]):
            raise ValueError(
                "traction_scale_fn must return one finite scalar, "
                f"got shape {np.asarray(scale).shape!r}."
            )
        return float(arr[0])

    def _assemble_traction_bc_rhs(self):
        """Assemble the natural traction contribution into the momentum rows.

        The dimensional traction ``t`` has units of stress, so the
        nondimensional traction entering the assembled CG weak form is

        ``t_d = t / Sigma_scale``.

        Callables are evaluated on dimensional coordinates flattened to
        ``(dim, n_points)``. Constant-array BCs normalized by the upstream
        ``_normalize_bc`` path also work through the same flattened interface.
        """
        traction_bcs = self.bc.get("t", {})
        if not traction_bcs:
            self._R_traction = np.zeros(self.ndofs, dtype=float)
            self._traction_scale_t0 = self._evaluate_traction_scale(0.0)
            return

        np_dofs = int(self.basis_p.N)
        nu_dofs = int(self.basis_u.N)
        scale = float(self.Sigma_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Sigma_scale must be positive, got {scale!r}.")

        self._R_traction = np.zeros(self.ndofs, dtype=float)
        for tag, traction_fun in traction_bcs.items():
            facets = self._boundary_facets_for_side(tag)
            if facets.size == 0:
                if self.verbose:
                    print(f"  Warning: traction boundary tag {tag!r} not found in mesh")
                continue

            fb = FacetBasis(self.mesh, self.el_u, facets=facets, intorder=self.intorder)

            @LinearForm
            def traction_form(v, w):
                x_dim = np.asarray(w.x, dtype=float) * float(self.L)
                x_flat = x_dim.reshape(self.dim, -1)
                raw = np.asarray(traction_fun(x_flat), dtype=float)

                if raw.ndim == 0:
                    raw = np.full((self.dim, x_flat.shape[1]), float(raw), dtype=float)
                elif raw.ndim == 1:
                    if raw.size == self.dim:
                        raw = np.broadcast_to(raw.reshape(self.dim, 1), (self.dim, x_flat.shape[1]))
                    elif raw.size == x_flat.shape[1]:
                        raw = np.broadcast_to(raw.reshape(1, x_flat.shape[1]), (self.dim, x_flat.shape[1]))
                    else:
                        raise ValueError(
                            f"traction BC on {tag!r} returned shape {raw.shape}; "
                            f"expected scalar, ({self.dim},), or ({self.dim}, n_points)."
                        )
                elif raw.shape[0] != self.dim:
                    raise ValueError(
                        f"traction BC on {tag!r} returned shape {raw.shape}; "
                        f"expected leading dimension {self.dim}."
                    )

                target_shape = (self.dim,) + tuple(np.asarray(w.x).shape[1:])
                if raw.shape == target_shape:
                    t_dim = raw
                elif raw.shape == (self.dim, x_flat.shape[1]):
                    t_dim = raw.reshape(target_shape)
                else:
                    t_dim = np.broadcast_to(raw, target_shape)

                t_nd = t_dim / scale
                return sum(t_nd[i] * v[i] for i in range(self.dim))

            bc_vec = traction_form.assemble(fb)
            self._R_traction[np_dofs : np_dofs + nu_dofs] += np.asarray(
                bc_vec,
                dtype=float,
            ).ravel()

        self._traction_scale_t0 = self._evaluate_traction_scale(0.0)
        self.R0 += self._traction_scale_t0 * self._R_traction

    def _traction_rhs_delta(self, t):
        """Return the extra RHS contribution from time-varying traction scaling."""
        if self._R_traction is None:
            return None
        T = float(self.get_scales()[0])
        t_dim = float(t) * T
        scale_now = self._evaluate_traction_scale(t_dim)
        delta_scale = scale_now - float(self._traction_scale_t0)
        if abs(delta_scale) <= 0.0:
            return None
        delta = delta_scale * self._R_traction
        if hasattr(self, "_dirichlet_dof_set"):
            delta = np.asarray(delta, dtype=float).copy()
            delta[np.asarray(self._dirichlet_dof_set, dtype=int)] = 0.0
        return delta

    def rhs(self, t, y, q_total):
        """Add time-varying traction scaling on top of the base CG rhs."""
        out = np.asarray(super().rhs(t, y, q_total), dtype=float)
        delta = self._traction_rhs_delta(t)
        if delta is not None:
            out = out + delta
        return out

    def solve_steady(self, q_total=None, t=0.0):
        """Respect the traction scaling function in steady solves as well."""
        y = super().solve_steady(q_total=q_total, t=t)
        return y

    def _boundary_facets_for_side(self, side: str) -> np.ndarray:
        """Return boundary facets for a coordinate-aligned side tag."""
        boundary_facets = np.asarray(self.mesh.boundary_facets(), dtype=int).ravel()
        if boundary_facets.size == 0:
            return boundary_facets

        facet_nodes = self.mesh.facets[:, boundary_facets]
        facet_midpoints = np.mean(self.mesh.p[:, facet_nodes], axis=1)
        xmin = float(np.min(self.mesh.p[0]))
        xmax = float(np.max(self.mesh.p[0]))
        ymin = float(np.min(self.mesh.p[1]))
        ymax = float(np.max(self.mesh.p[1]))
        tol = 1.0e-10

        side_norm = str(side).strip().lower()
        if side_norm == "left":
            mask = np.abs(facet_midpoints[0] - xmin) < tol
        elif side_norm == "right":
            mask = np.abs(facet_midpoints[0] - xmax) < tol
        elif side_norm == "bottom":
            mask = np.abs(facet_midpoints[1] - ymin) < tol
        elif side_norm == "top":
            mask = np.abs(facet_midpoints[1] - ymax) < tol
        else:
            return np.zeros(0, dtype=int)
        return boundary_facets[np.asarray(mask, dtype=bool)]
