"""Injection-only diagnostic: Biot poroelastodynamics, no friction.

Verifies that the pressure source at point_source_loc actually reaches the
right (fault) boundary within the simulation horizon. No NCP contact.

Run:
    python _poro_injection_only.py [Q0_SCALE] [n_steps]

Env overrides:
    PORO_Q0=<float>  amplitude override (default 1e-7)
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skfem import (
    Basis, ElementVector, ElementComposite, FacetBasis,
    ElementLineP2, ElementLineP1, MeshLine,
)
from skfem.assembly import asm, BilinearForm, LinearForm
from skfem.helpers import grad, dot, ddot

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import block_diag as sp_block_diag, vstack as sp_vstack
from scipy.sparse.linalg import splu

import autograd.numpy as anp
from autograd import elementwise_grad

from scipy.integrate import solve_ivp as scipy_solve_ivp


# ── Physical parameters (mirrors notebook) ──────────────────────────────
L = 1e-3
ρf = 1000 * L**3
ρs = 2500 * L**3
Kf = 2.2e9 * L
Ks = 80e9 * L
nu = .15
φ = .2
α = .6
turt = 3.2

k_vv = 1e18 * L**3
k_vu = 1e18 * L**3
k_r = 1e21 * L**3
ηs = 9e-2

k = 1e-18 * L**-2
μf = 1e-3 * L

Kb = Ks * (1 - α)
G = 3/2 * Kb * (1 - 2*nu) / (1 + nu)
M = ((α - φ)/Ks + φ/Kf) ** -1
ρ11 = (1 - φ)*ρs + (turt - 1)*φ*ρf
ρ22 = turt * φ * ρf
ρ12 = -(turt - 1) * φ * ρf
mu_lame = G
lam = Kb - 2*G/3
b = (k / μf) ** -1

robin_left  = (k_vv, k_vv, k_vu, k_vu, k_r)
robin_right = (k_vv, 0,    k_vu, 0,    k_r)

D = 1000.
xmin, xmax = -D/2, D/2
xmin_d, xmax_d = xmin/L, xmax/L

Ku = Kb + α**2 * M
rho_b = (1 - φ)*ρs + φ*ρf
vw = np.sqrt((Ku + 4/3*mu_lame) / rho_b)
print(f"vw (undrained wave) = {vw * L:.3e} m/s")
print(f"wave transit L/2 / vw = {(D/2)/vw / L:.3f} s")

chy = k / μf / M**-1
print(f"tch_diff = {(D/L/2)**2 / chy:.3e} s  (source->fault distance = {(D/2)*0.2/L} non-dim)")


# ── Source config ───────────────────────────────────────────────────────
Q0_scale = float(os.environ.get('PORO_Q0', '1e-7'))
if len(sys.argv) > 1:
    Q0_scale = float(sys.argv[1])
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

point_source_loc = np.array([[(D/2) * 0.8 / L]])   # non-dim, 80% toward right fault
point_source_Q0  = np.array([[-Q0_scale]])

def smoothstep_quintic(x):
    return x*x*x*(x*(x*6 - 15) + 10)

def smooth_step(t, t0, tau):
    x = (t - t0) / tau
    x = anp.clip(x, 0.0, 1.0)
    return smoothstep_quintic(x)

def smooth_pulse(t, t_start, t_end, tau=1e-4):
    return (1 - smooth_step(t, t_end, tau)) * smooth_step(t, t_start, tau)

dsmooth_pulse = elementwise_grad(smooth_pulse, argnum=0)

def qv(t, t_start, t_end, tau=1e-4):
    return point_source_Q0 * smooth_pulse(t, t_start, t_end, tau) / L

def dqv(t, t_start, t_end, tau=1e-4):
    return point_source_Q0 * dsmooth_pulse(t, t_start, t_end, tau) / L


# ── Mesh ────────────────────────────────────────────────────────────────
mesh_elements = 40 + 1
xcoords = np.linspace(xmin_d, xmax_d, mesh_elements + 1)
bnd = {
    'left':  lambda x: np.isclose(x[0], xmin_d),
    'right': lambda x: np.isclose(x[0], xmax_d),
}
mesh = MeshLine(xcoords).with_boundaries(bnd)
left_facets  = mesh.boundaries['left']
right_facets = mesh.boundaries['right']


# ── FEM ─────────────────────────────────────────────────────────────────
def sij_core(E_kl):
    dim = E_kl.shape[-1]
    batch_shape = E_kl.shape[:-2]
    E3 = np.zeros(batch_shape + (3, 3), dtype=E_kl.dtype)
    E3[..., :dim, :dim] = E_kl
    trE = np.trace(E3, axis1=-2, axis2=-1)[..., None, None]
    I = np.eye(3, dtype=E3.dtype)
    s3 = 2.0 * mu_lame * E3 + lam * trE * I
    return s3[..., :dim, :dim]

def Sij(E):
    E_last = np.moveaxis(E, (0, 1), (-2, -1))
    S_last = sij_core(E_last)
    return np.moveaxis(S_last, (-2, -1), (0, 1))

def make_sym(gradu):
    o = np.zeros((2, 2, *gradu.shape[2:]), dtype=gradu.dtype)
    o[0, 0] = gradu[0, 0]
    o[0, 1] = 0.5 * gradu[1, 0]
    o[1, 0] = 0.5 * gradu[1, 0]
    return o

def my_sym_grad(u):
    return make_sym(grad(u))

dim = 2
intorder = 4
el_v = ElementVector(ElementLineP2(), dim=dim)
el_r = ElementVector(ElementLineP2(), dim=dim)
el_u = ElementVector(ElementLineP2(), dim=dim)
el_p = ElementLineP1()
mixed = ElementComposite(el_v, el_r, el_u, el_p)
basis = Basis(mesh, mixed, intorder=intorder)
fbasis_L = FacetBasis(mesh, mixed, intorder=intorder, facets=left_facets)
fbasis_R = FacetBasis(mesh, mixed, intorder=intorder, facets=right_facets)

@BilinearForm
def varform_lhs(dv, dr, du, dp, δv, δr, δu, δp, w):
    return (ρ11 * dot(dv, δv) + ρ22 * dot(dr, δr)
            + ρ12 * dot(dv, δr) + ρ12 * dot(dr, δv)
            + dot(du, δu) + dp * δp)

@BilinearForm
def varform_rhs(v, r, u, p, δv, δr, δu, δp, w):
    Eps = my_sym_grad(u)
    Su = Sij(Eps)
    Sv = ηs * Sij(my_sym_grad(v))
    I = np.eye(Eps.shape[0])[:, :, None, None] * np.ones_like(Eps)
    S = Su + Sv - α * I * p
    t1 = + ddot(S, my_sym_grad(δv)) + b * dot(r, δr) - dot(v, δu)
    t2 = - dot(p, grad(δr)[0, 0]) + M * dot(α * grad(v)[0, 0] + grad(r)[0, 0], δp)
    return t1 + t2

@BilinearForm
def varform_robin(v, r, u, p, δv, δr, δu, δp, w):
    c_vv1, c_vv2, c_vu1, c_vu2, c_r = w.params
    t1 = (c_vv1 * v[0] + c_vu1 * u[0]) * δv[0]
    t1 += (c_vv2 * v[1] + c_vu2 * u[1]) * δv[1]
    t2 = c_r * r[0] * δr[0]
    return t1 + t2

E = asm(varform_lhs, basis)
A = -asm(varform_rhs, basis)
A -= asm(varform_robin, fbasis_L, params=robin_left)
A -= asm(varform_robin, fbasis_R, params=robin_right)
ndofs = basis.N
print(f"ndofs = {ndofs}")


# ── Source matrix (smoothed Dirac on pressure DOF) ──────────────────────
dx_mesh = (np.amax(mesh.p) - np.amin(mesh.p)) / mesh_elements
sigma = 1.5 * dx_mesh
print(f"sigma (source Gaussian) = {sigma:.3f}  non-dim   (dx = {dx_mesh:.3f})")
print(f"source → fault distance = {xmax_d - point_source_loc[0, 0]:.3f}  "
      f"({(xmax_d - point_source_loc[0, 0]) / sigma:.2f}σ)")

nsrc = len(point_source_loc)
Bp = lil_matrix((ndofs, nsrc))
for j, c in enumerate(point_source_loc):
    @LinearForm
    def smooth_dirac(δv, δr, δu, δp, w):
        r2 = np.sum((w.x - c[:, None, None])**2, axis=0)
        g = np.exp(-r2 / 2 / sigma**2)
        norm = (sigma * np.sqrt(2 * np.pi)) ** w.x.shape[0]
        return g / norm * δp / dx_mesh
    fvec = asm(smooth_dirac, basis)
    Bp[:, j] = fvec.reshape(-1, 1) / L
B_src = (M * Bp).tocsr()


# ── Dirichlet BCs (u1, u2 pinned on left only — no friction, free right wall) ──
user_to_skfem = {
    "v1": "u^1^1", "v2": "u^2^1",
    "r1": "u^1^2", "r2": "u^2^2",
    "u1": "u^1^3", "u2": "u^2^3",
    "p":  "u^4",
}

bc_dofs = {'u1': {'left': 0.}, 'u2': {'left': 0.}}

E_D = E.tolil(); A_D = A.tolil(); B_D = B_src.tolil()
dofs_D = np.array([], dtype=np.int32)
for field, sides_bc in bc_dofs.items():
    for side in sides_bc:
        dofs_D = np.append(dofs_D, basis.get_dofs(side).all(user_to_skfem[field]))
E_D[dofs_D, :] = 0.
E_D[:, dofs_D] = 0.
E_D[dofs_D, dofs_D] = 1.
A_D[dofs_D, :] = 0.
A_D[:, dofs_D] = 0.
B_D[dofs_D] = 0.
E_D = E_D.tocsr(); A_D = A_D.tocsr(); B_D = B_D.tocsr()


# ── Right-boundary DOF indices (for diagnostics) ────────────────────────
def get_right_idx():
    out = {}
    for f, sk in user_to_skfem.items():
        out[f] = int(basis.get_dofs('right').all(sk)[0])
    return out
idxR = get_right_idx()
print(f"right-boundary dof indices: {idxR}")


# ── RHS for scipy solve_ivp (descriptor: E y' = A y + B dq(t)) ──────────
from scipy.sparse.linalg import splu as _splu
E_lu = _splu(E_D.tocsc())

load_t_end = 5.0
tau_ld = 0.1
def dq(t):
    return dqv(t, 0.0, load_t_end, tau_ld).reshape(-1)

def f_ode(t, y):
    rhs = A_D @ y + B_D @ dq(t)
    return E_lu.solve(rhs)

y0 = np.zeros(ndofs)
tmax = load_t_end
print(f"\nRunning scipy solve_ivp BDF, tmax={tmax}, Q0_scale={Q0_scale}")
start = time.time()
sol = scipy_solve_ivp(
    f_ode, (0.0, tmax), y0,
    method='BDF', rtol=1e-6, atol=1e-10,
    t_eval=np.linspace(0, tmax, 501),
)
print(f"elapsed: {time.time()-start:.1f} s   success={sol.success}  nfev={sol.nfev}")
print(f"  message: {sol.message}")


# ── Extract fields ──────────────────────────────────────────────────────
def split_sol(zs):
    (v, vb), (r, rb), (u, ub), (p, pb) = basis.split(zs)
    v = [vi[vib.nodal_dofs[0]] for vi, vib in vb.split(v)]
    r = [ri[rib.nodal_dofs[0]] for ri, rib in rb.split(r)]
    u = [ui[uib.nodal_dofs[0]] for ui, uib in ub.split(u)]
    p = p[pb.nodal_dofs[0]]
    return np.array([*v[:2], *r[:2], *u[:2], p])

ys = sol.y
ts = sol.t
print(f"time samples: {ts.size}")

# Pressure at source, center, fault
p_ids = basis.get_dofs().all('u^4')  # all pressure DOFs — use nodal ones
p_dof_src  = idxR['p']  # fault
# Find closest-to-source mesh node pressure DOF
p_basis_nodes = mesh.p[0]
src_loc = point_source_loc[0, 0]
# scalar p basis nodal indices
(_, _), (_, _), (_, _), (p_, pbasis) = basis.split(np.zeros(ndofs))
p_nodal = pbasis.nodal_dofs[0]  # global DOF ids
p_xcoords = mesh.p[0, :]  # node x positions
p_idx_src = int(p_nodal[np.argmin(np.abs(p_xcoords - src_loc))])
p_idx_cen = int(p_nodal[np.argmin(np.abs(p_xcoords - 0.0))])
p_idx_flt = int(p_nodal[np.argmin(np.abs(p_xcoords - xmax_d))])

p_src = ys[p_idx_src] / L
p_cen = ys[p_idx_cen] / L
p_flt = ys[p_idx_flt] / L
v2_flt = ys[idxR['v2']] * L
u2_flt = ys[idxR['u2']] * L


print("\n=== Pressure time histories (physical units) ===")
print(f"  source location (nondim): x={src_loc:.1f}  (fault at x={xmax_d:.1f})")
print(f"  source p range : [{p_src.min():.3e}, {p_src.max():.3e}]")
print(f"  center p range : [{p_cen.min():.3e}, {p_cen.max():.3e}]")
print(f"  fault  p range : [{p_flt.min():.3e}, {p_flt.max():.3e}]")
print(f"  |v2| at fault max : {np.max(np.abs(v2_flt)):.3e} m/s")
print(f"  |u2| at fault max : {np.max(np.abs(u2_flt)):.3e} m")

s11_eff_0 = 4*1e6 * L
print(f"\n  compare to s11_eff_0 = {s11_eff_0:.3e}  (cone ray in notebook)")
print(f"  ratio p_fault_max / s11_eff_0 = {p_flt.max() / s11_eff_0:.3e}")


# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)

axes[0, 0].plot(ts, p_src, label='source')
axes[0, 0].plot(ts, p_cen, label='center')
axes[0, 0].plot(ts, p_flt, label='fault')
axes[0, 0].set_xlabel('t [s]'); axes[0, 0].set_ylabel('p [Pa·L]')
axes[0, 0].legend(); axes[0, 0].grid(True)
axes[0, 0].set_title(f'Pressure histories (Q0={Q0_scale:.1e})')

axes[0, 1].plot(ts, v2_flt, 'C3')
axes[0, 1].set_xlabel('t [s]'); axes[0, 1].set_ylabel('v2 fault [m/s]')
axes[0, 1].grid(True); axes[0, 1].set_title('Tangential velocity at fault')

axes[1, 0].plot(ts, u2_flt, 'C3')
axes[1, 0].set_xlabel('t [s]'); axes[1, 0].set_ylabel('u2 fault [m]')
axes[1, 0].grid(True); axes[1, 0].set_title('Tangential displacement at fault')

# Spatial profile at t=tmax/2 and tmax
for t_frac, col in [(0.25, 'C0'), (0.5, 'C1'), (1.0, 'C3')]:
    ti = np.argmin(np.abs(ts - t_frac * tmax))
    z = split_sol(ys[:, ti])
    axes[1, 1].plot(xcoords * L, z[6] / L, color=col,
                    label=f't={ts[ti]:.2f} s')
axes[1, 1].axvline(src_loc * L, color='k', ls=':', lw=0.6, label='source')
axes[1, 1].axvline(xmax_d * L, color='r', ls=':', lw=0.6, label='fault')
axes[1, 1].set_xlabel('x [m]'); axes[1, 1].set_ylabel('p [Pa·L]')
axes[1, 1].legend(); axes[1, 1].grid(True)
axes[1, 1].set_title('Pressure spatial profile')

plt.tight_layout()
out = f'/tmp/poro_inject_only_Q0_{Q0_scale:.0e}.png'
plt.savefig(out, dpi=110)
print(f"\nwrote {out}")
