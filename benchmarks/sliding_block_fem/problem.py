"""Many-DOF 2D elastic bouncing-block benchmark.

Rectangular elastic block dropped from a height onto a rigid floor at y=0.
Pure gravity, no traction. Bottom face carries unilateral Signorini contact
with Coulomb friction (friction is inactive in vertical-only motion but the
tangential constraint exercises the full contact machinery).

State: y = [v (2*Nn), u (2*Nn)]
Descriptor: A = block_diag(M, I)
Smooth rhs: [-K u + f_grav;  v]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from skfem import (
    MeshQuad,
    ElementVector,
    ElementQuad1,
    Basis,
    BilinearForm,
    LinearForm,
    asm,
)
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity, lame_parameters


@dataclass
class BouncingBlockProblem:
    mesh: object
    basis: object
    K: sp.csr_matrix
    M: sp.csr_matrix
    A: sp.csr_matrix
    rhs_smooth: Callable
    rhs_jac: Callable
    y0: np.ndarray
    contacts: list
    gap_func: Callable
    bottom_nodes: np.ndarray
    bottom_node_x: np.ndarray
    bottom_node_y_ref: np.ndarray   # reference y-coord of bottom nodes
    mu: float
    g: float
    n_phys: int
    n_dof: int
    n_nodes: int
    L: float
    H: float
    drop_height: float
    component_slices: list


def build(nx: int = 20, ny: int = 10,
          L: float = 4.0, H: float = 1.0,
          E: float = 1.0e4, nu: float = 0.3, rho: float = 1.0,
          g: float = 9.81, mu_fric: float = 0.0,
          drop_height: float = 0.5,
          lumped_mass: bool = True) -> BouncingBlockProblem:
    mesh = MeshQuad.init_tensor(
        np.linspace(0.0, L, nx + 1),
        np.linspace(drop_height, drop_height + H, ny + 1),
    )
    elem = ElementVector(ElementQuad1())
    basis = Basis(mesh, elem)

    lam, mu_lame = lame_parameters(E, nu)
    K = asm(linear_elasticity(lam, mu_lame), basis).tocsr()

    @BilinearForm
    def mass_form(u, v, w):
        return rho * dot(u, v)

    M = asm(mass_form, basis).tocsr()
    if lumped_mass:
        M = sp.diags(np.asarray(M.sum(axis=1)).ravel(), format="csr")

    n_dof = basis.N
    n_nodes = mesh.p.shape[1]
    assert n_dof == 2 * n_nodes

    @LinearForm
    def grav_form(v, w):
        return -rho * g * v.value[1]

    f_grav = asm(grav_form, basis)

    # Bottom-face (y == drop_height) nodes, ordered left-to-right
    on_bottom = np.isclose(mesh.p[1], drop_height)
    bottom_node_ids = np.where(on_bottom)[0]
    bottom_order = np.argsort(mesh.p[0, bottom_node_ids])
    bottom_nodes = bottom_node_ids[bottom_order]
    bottom_x = mesh.p[0, bottom_nodes]
    bottom_y_ref = mesh.p[1, bottom_nodes]

    ux_idx = 2 * bottom_nodes
    uy_idx = 2 * bottom_nodes + 1

    n_phys = 2 * n_dof
    component_slices = [slice(0, n_dof), slice(n_dof, 2 * n_dof)]

    A = sp.block_diag([M, sp.eye(n_dof, format="csr")], format="csr").tocsr()

    rhs_buf = np.zeros(n_phys)

    def rhs_smooth(t, y):
        u = y[n_dof:]
        v = y[:n_dof]
        rhs_buf[:n_dof] = -(K @ u) + f_grav
        rhs_buf[n_dof:] = v
        return rhs_buf.copy()

    J_const = sp.bmat(
        [[sp.csr_matrix((n_dof, n_dof)), -K],
         [sp.eye(n_dof, format="csr"),   sp.csr_matrix((n_dof, n_dof))]],
        format="csr",
    )

    def rhs_jac(t, y, Fk_val=None):
        return J_const

    contacts = [
        {
            "vel_normal_idx": int(uy_idx[k]),
            "vel_tangential_idx": int(ux_idx[k]),
            "mu": float(mu_fric),
        }
        for k in range(len(bottom_nodes))
    ]

    def gap_func(y_phys, t):
        # Signed distance of bottom nodes to floor at y=0.
        return bottom_y_ref + y_phys[n_dof + uy_idx]

    y0 = np.zeros(n_phys)

    return BouncingBlockProblem(
        mesh=mesh, basis=basis, K=K, M=M, A=A,
        rhs_smooth=rhs_smooth, rhs_jac=rhs_jac, y0=y0,
        contacts=contacts, gap_func=gap_func,
        bottom_nodes=bottom_nodes,
        bottom_node_x=bottom_x,
        bottom_node_y_ref=bottom_y_ref,
        mu=mu_fric, g=g,
        n_phys=n_phys, n_dof=n_dof, n_nodes=n_nodes,
        L=L, H=H, drop_height=drop_height,
        component_slices=component_slices,
    )


if __name__ == "__main__":
    p = build(nx=5, ny=3)
    print(f"n_nodes={p.n_nodes}  n_phys={p.n_phys}  n_contacts={len(p.contacts)}")
    print(f"drop_height={p.drop_height}  gap(y0)={p.gap_func(p.y0, 0.0)}")
    r0 = p.rhs_smooth(0.0, p.y0)
    print(f"||rhs(0, y0)||_inf = {np.max(np.abs(r0)):.3e}")
