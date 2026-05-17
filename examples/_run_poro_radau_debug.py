"""Standalone script extracted from porodynamics_radau_friction notebook for debugging."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skfem import Basis, ElementVector, ElementComposite, FacetBasis
from skfem.assembly import asm, BilinearForm, LinearForm
from skfem.helpers import grad, sym_grad, div, dot, ddot, trace, eye
from skfem import MeshLine, ElementLineP2, ElementLineP1
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
import time

import autograd.numpy as anp
from autograd import elementwise_grad

import solve_nivp
from solve_nivp.contact import build_impulse_contact

# ── Parameters ──────────────────────────────────────────────────────────
L = 1e-3

ρf = 1000 * L**3
ρs = 2500 * L**3

Kf = 2.2e9 * L**1
Ks = 80e9 * L**1
nu = .15
φ = .2
α = .6

turt = 1

k_vv = 1e18*L**3
k_vu = 1e18*L**3
k_r = 1e21*L**3
ηs_override = float(sys.argv[3]) if len(sys.argv) > 3 else 9e-2
ηs = ηs_override

k = 1e-18 * L**-2
μf = 1e-3 * L**1

Kb = Ks*(1-α)
G = 3/2*Kb*(1-2*nu)/(1+nu)
M = ((α-φ)/Ks + φ/Kf)**-1

ρ11 = (1 - φ) * ρs + (turt - 1) * φ * ρf
ρ22 = turt * φ * ρf
ρ12 = -(turt - 1) * φ * ρf
Δ = ρ11*ρ22 - ρ12**2
mu = G
lam = Kb - 2*G/3
b = (k / μf)**-1

robin_left = (k_vv, k_vv, k_vu, k_vu, k_r)
robin_right = (k_vv, 0, k_vu, 0, k_r)

bc_dofs = {
    'u1': {'left': 0.},
    'u2': {'left': 0.},
}

Dc=.10 /1000/L
Dmu=-.1
mu_res=.5
s11_eff_0 = 4*1e6 * L**1
friction_sides = ["right"]

D = 1000.

Ku = Kb + α**2*M
rho_b = (1-φ)*ρs + φ*ρf
vw = np.sqrt((Ku + 4/3*mu)/rho_b)
print("Δ:", Δ)
print("α:", α)
print("Kb:", Kb*1e-9*L**-1)
print("G:", G*1e-9*L**-1)

Cm = np.array([[Kb + 4/3*G, α*M],[α*M, M]])
Rm = np.array([[ρ11, ρ12],[ρ12, ρ22]])
tmp = np.linalg.solve(Rm, Cm)
c2, vecs = np.linalg.eig(tmp)
vw1, vw2 = np.sqrt(c2)

print("wave velocities", vw1 * L**1, vw2 * L**1)
L_bar = D / L
k_eff = G / L_bar
m_eff = ρ11 * L_bar / 2
omega0 = np.sqrt(k_eff / m_eff)
zeta = ηs * k_eff / (2 * np.sqrt(k_eff * m_eff))

EQslip = k_eff < (-Dmu * s11_eff_0 / Dc)
expd_slip_static = -Dmu * s11_eff_0 / G * L_bar

if EQslip and zeta < 1:
    overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    expd_slip = expd_slip_static * (1 + overshoot)
else:
    expd_slip = expd_slip_static

print("k_eff:", k_eff, "  weakening rate:", -Dmu * s11_eff_0 / Dc)
print("EQ-like slip:", EQslip)
print(f"    omega0={omega0:.3f} rad/s,  zeta={zeta:.4f}")
print(f"    Static equilibrium     : {expd_slip_static * L:.6e} m")
print(f"    Damped arrest (analyt.): {expd_slip * L:.6e} m")
print(f"    Overshoot factor       : {expd_slip / expd_slip_static:.3f}")

# ── Injection source ────────────────────────────────────────────────────
ENABLE_INJECTION = bool(int(__import__('os').environ.get('POROINJ', '0')))
point_source_loc = np.array([[(D / 2) * 0.8 / L]])  # 80% toward right fault (non-dim)
point_source_Q0 = np.array([[-1e-5]])                    # amplitude for O(s11_eff_0) pressure

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
    q0 = point_source_Q0
    return q0[:] * smooth_pulse(t, t_start, t_end, tau)/L**1

def dqv(t, t_start, t_end, tau=1e-4):
    q0 = point_source_Q0
    return q0[:] * dsmooth_pulse(t, t_start, t_end, tau)/L**1

tmax_plot = 1.
tau = .1
q = lambda t: qv(t, t_start=0, t_end=5*tmax_plot/3, tau=tau).squeeze()
dq = lambda t: dqv(t, t_start=0, t_end=5*tmax_plot/3, tau=tau).squeeze()

# ── Mesh ────────────────────────────────────────────────────────────────
xmin, xmax = -D/2, D/2
xmin_d, xmax_d = xmin/L, xmax/L

bnd_labels = {
    'left': lambda x: np.isclose(x[0], xmin_d),
    'right': lambda x: np.isclose(x[0], xmax_d),
}

mesh_elements = 40+1
xcoords = np.linspace(xmin_d, xmax_d, mesh_elements + 1)
mesh = MeshLine(xcoords)
mesh = mesh.with_boundaries(bnd_labels)
left_facets = mesh.boundaries['left']
right_facets = mesh.boundaries['right']

# ── FEM ─────────────────────────────────────────────────────────────────
def sij_core(E_kl, x_type):
    dim = E_kl.shape[-1]
    if dim == 2:
        batch_shape = E_kl.shape[:-2]
        E_kl_3D = np.zeros(batch_shape + (3, 3), dtype=E_kl.dtype)
        E_kl_3D[..., :dim, :dim] = E_kl
    elif dim == 3:
        E_kl_3D = E_kl
    else:
        raise ValueError("dim must be 1, 2 or 3")
    trE = np.trace(E_kl_3D, axis1=-2, axis2=-1)[..., None, None]
    I = np.eye(3, dtype=E_kl_3D.dtype)
    s3 = 2.0 * mu * E_kl_3D + lam * trE * I
    return s3[..., :dim, :dim]

def Sij(E, x_type):
    E = np.asarray(E)
    if E.ndim < 2 or E.shape[0] != E.shape[1] or E.shape[0] not in (2, 3):
        raise ValueError(f"Expected (2,2,...) or (3,3,...), got {E.shape}")
    E_last = np.moveaxis(E, (0, 1), (-2, -1))
    S_last = sij_core(E_last, x_type)
    return np.moveaxis(S_last, (-2, -1), (0, 1))

dim = 2
intorder = 4

element_v = ElementLineP2()
element_r = ElementLineP2()
element_u = ElementLineP2()
element_p = ElementLineP1()

el_v = ElementVector(element_v, dim=dim)
el_r = ElementVector(element_r, dim=dim)
el_u = ElementVector(element_u, dim=dim)
el_p = element_p

mixed_element = ElementComposite(el_v, el_r, el_u, el_p)
basis = Basis(mesh, mixed_element, intorder=intorder)
basis_boundary = FacetBasis(mesh, mixed_element, intorder=intorder)
fbasis_left = FacetBasis(mesh, mixed_element, intorder=intorder, facets=left_facets)
fbasis_right = FacetBasis(mesh, mixed_element, intorder=intorder, facets=right_facets)

@BilinearForm
def varform_lhs(dv, dr, du, dp, δv, δr, δu, δp, w):
    return ρ11 * dot(dv, δv) + ρ22 * dot(dr, δr) + ρ12 * dot(dv, δr) + ρ12 * dot(dr, δv) + dot(du, δu) + dp * δp

def make_sym(gradu):
    o = np.zeros((2, 2, *gradu.shape[2:]), dtype=gradu.dtype)
    o[0, 0] = gradu[0, 0]
    o[0, 1] = 0.5 * gradu[1, 0]
    o[1, 0] = 0.5 * gradu[1, 0]
    return o

def my_sym_grad(u):
    tt = grad(u)
    return make_sym(tt)

@BilinearForm
def varform_rhs(v, r, u, p, δv, δr, δu, δp, w):
    Eps_kl = my_sym_grad(u)
    Su_ij = Sij(Eps_kl, "quad")
    dEps_kl = my_sym_grad(v)
    Sv_ij = ηs * Sij(dEps_kl, "quad")
    I = np.eye(Eps_kl.shape[0])[:, :, None, None] * np.ones_like(Eps_kl)
    S_ij = Su_ij + Sv_ij - α * I * p
    term_1 = + ddot(S_ij, my_sym_grad(δv)) + b * dot(r, δr) - dot(v, δu)
    term_2 = - dot(p, grad(δr)[0,0]) + M * dot(α * grad(v)[0,0] + grad(r)[0,0], δp)
    return term_1 + term_2

@BilinearForm
def varform_robin(v, r, u, p, δv, δr, δu, δp, w):
    c_vv1, c_vv2, c_vu1, c_vu2, c_r = w.params
    rn = r[0]
    δrn = δr[0]
    term1 = (c_vv1 * v[0] + c_vu1 * u[0]) * δv[0]
    term1+= (c_vv2 * v[1] + c_vu2 * u[1]) * δv[1]
    term2 = c_r * rn * δrn
    return term1 + term2

@BilinearForm
def varform_frictional_part(v, r, u, p, δv, δr, δu, δp, w):
    n = w.n
    Eps_kl = my_sym_grad(u)
    Su_ij = Sij(Eps_kl, "quad")
    dEps_kl = my_sym_grad(v)
    Sv_ij = ηs * Sij(dEps_kl, "quad")
    I = np.eye(Eps_kl.shape[0])[:, :, None, None] * np.ones_like(Eps_kl)
    S_ij = Su_ij + Sv_ij - α * I * p
    s12_t = S_ij[0,1]
    t2 = s12_t * n
    return t2 * δv[1]

@BilinearForm
def varform_force(fv, fr, fu, fp, δv, δr, δu, δp, w):
    return + dot(fv + fu, δv) + dot(fr, δr)

E = asm(varform_lhs, basis)
Abulk = - asm(varform_rhs, basis)
Abndr_left = - asm(varform_robin, fbasis_left, params=robin_left)
Abndr_right = - asm(varform_robin, fbasis_right, params=robin_right)
A = Abulk + Abndr_left + Abndr_right

dofs_left  = basis.get_dofs('left').all(["u^1^1", "u^2^1", "u^1^2", "u^2^2"])
Bbndr_force_left = - asm(varform_force, fbasis_left)[:, dofs_left]
dofs_right = basis.get_dofs('right').all(["u^1^1", "u^2^1", "u^1^2", "u^2^2"])
Bbndr_force_right = - asm(varform_force, fbasis_right)[:, dofs_right]

ndofs = basis.N
print(f"ndofs = {ndofs}")

# ── Source matrix ───────────────────────────────────────────────────────
dx = (np.amax(mesh.p) - np.amin(mesh.p)) / mesh_elements
sources_RMS_width = 1.5 * dx
nsrc = len(point_source_loc)
Bp_full = lil_matrix((ndofs, nsrc))
sigma = sources_RMS_width
for j, (c_loc) in enumerate(point_source_loc):
    @LinearForm
    def smooth_dirac(δv, δr, δu, δp, w):
        r2 = np.sum((w.x - c_loc[:, None, None])**2, axis=0)
        g = np.exp(-r2 / 2 / sigma**2)
        norm_factor = (sigma * np.sqrt(2 * np.pi))**w.x.shape[0]
        return g / norm_factor * δp / dx
    fvec = asm(smooth_dirac, basis)
    Bp_full[:, j] = fvec.reshape(-1, 1)/L
B_src = M * Bp_full.tocsr()

# ── BC enforcement ──────────────────────────────────────────────────────
user_to_skfem = {
    "v1": "u^1^1", "v2": "u^2^1",
    "r1": "u^1^2", "r2": "u^2^2",
    "u1": "u^1^3", "u2": "u^2^3",
    "p": "u^4",
}

E_D = E.tolil(); A_D = A.tolil(); B_D = B_src.tolil()
dofs_D = np.array([], dtype=np.int32)
for field, sides_bc in bc_dofs.items():
    for side, value in sides_bc.items():
        field_skfem = user_to_skfem.get(field)
        dofs_D = np.append(dofs_D, basis.get_dofs(side).all(field_skfem))
E_D[dofs_D, :] = 0.
E_D[:, dofs_D] = 0.
E_D[dofs_D, dofs_D] = 1.
A_D[dofs_D, :] = 0.
A_D[:, dofs_D] = 0.
B_D[dofs_D] = 0.
E_D = E_D.tocsr(); A_D = A_D.tocsr(); B_D = B_D.tocsr()

# Augment with accumulated slip DOF
from scipy.sparse import block_diag as sp_block_diag, vstack as sp_vstack, csr_matrix as csr
slip_idx = ndofs
eps_slip = 1e-20 / L
slip_rate_floor = np.sqrt(eps_slip)

E_D = sp_block_diag([E_D, csr(np.array([[1.]]))], format='csr')
A_D = sp_block_diag([A_D, csr(np.array([[0.]]))], format='csr')
B_D = sp_vstack([B_D, csr((1, B_D.shape[1]))], format='csr')

ndofs_aug = ndofs + 1
print(f"Augmented system: {ndofs} physics + 1 slip = {ndofs_aug} DOFs")

# ── Boundary extraction ─────────────────────────────────────────────────
sides = ["left", "right"]
TPloc = {"left": 0, "right": 0}
TPloc_idx = {"left": 0, "right": 0}
for side in sides:
    t1_node = basis.get_dofs(side).all(user_to_skfem["v1"])[0]
    t2_node = basis.get_dofs(side).all(user_to_skfem["v2"])[0]
    v1_node = t1_node
    v2_node = t2_node
    u1_node = basis.get_dofs(side).all(user_to_skfem["u1"])[0]
    u2_node = basis.get_dofs(side).all(user_to_skfem["u2"])[0]
    p_node  = basis.get_dofs(side).all(user_to_skfem["p"])[0]
    TPloc_idx[side] = [t1_node, t2_node, v1_node, v2_node, u1_node, u2_node, p_node]

# ── RHS ─────────────────────────────────────────────────────────────────
_, _, vtn_idx, vtt_idx, u1n_idx, u2t_idx, pt_idx = TPloc_idx["right"]

def rhs_plant(t, y):
    rhs_sm = A_D @ y
    if ENABLE_INJECTION:
        rhs_sm = rhs_sm + B_D @ dq(t)
    v2 = y[vtt_idx]
    rhs_sm[slip_idx] = np.sqrt(v2**2 + eps_slip) - slip_rate_floor
    return rhs_sm

# ── Friction ────────────────────────────────────────────────────────────
def mu_fric(slip, slip_rate):
    a = mu_res * (1. - Dmu/mu_res * np.exp(-slip/Dc))
    return a

def mu_(z):
    _, _, vtn_idx_, vtt_idx_, u1n_idx_, u2t_idx_, _ = TPloc_idx["right"]
    slip_rate = z[vtt_idx_]
    slip = z[slip_idx]
    return mu_fric(slip, slip_rate)

# ── Initial condition ───────────────────────────────────────────────────
def IC(xn):
    x = xn[0] * L
    v = 1.e-6 * (x + D / 2) / D
    return v / L

y0 = np.zeros(ndofs)
(vi, vb), (ri, rb), (ui, ub), (pi, pb) = basis.split(y0)
(v1i, vb1), (v2i, vb2) = vb.split(y0)
idx2 = vb.split_indices()[1]
v20 = vb1.project(IC)
vi[idx2] = v20
idx = basis.split_indices()[0]
y0[idx] = vi
y0 = np.append(y0, 0.0)

# ── Run config ──────────────────────────────────────────────────────────
USE_ADAPTIVE = False
tmax = 5.0
load_t_end = tmax

# Configurable from command line: python _run_poro_radau_debug.py [n_steps] [method]
import sys
n_steps_init = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
METHOD = sys.argv[2] if len(sys.argv) > 2 else 'radau_iia'
t_span = (0.0, tmax)
h0_init = tmax / n_steps_init

# Contact frame
contact_vel_C = lil_matrix((2, ndofs_aug), dtype=float)
contact_vel_C[0, vtn_idx] = 1.0
contact_vel_C[1, vtt_idx] = 1.0
contact_vel_C = contact_vel_C.tocsr()

contact_gap_C = lil_matrix((2, ndofs_aug), dtype=float)
contact_gap_C[0, u1n_idx] = -1.0
contact_gap_C[1, u2t_idx] = 1.0
contact_gap_C = contact_gap_C.tocsr()

contact_B = contact_vel_C.T.tocsr()

contact_blocks = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=mu_)]
contact_vel_idx = np.array([vtn_idx, vtt_idx], dtype=int)
contact_other_idx = np.array(
    sorted(set(range(ndofs_aug)) - set(contact_vel_idx.tolist())),
    dtype=int,
)
contact_component_slices = [contact_vel_idx, contact_other_idx]

# Smoothed loading pulse
dx_mesh = (np.amax(mesh.p) - np.amin(mesh.p)) / mesh_elements
dt_crit = dx_mesh / vw
tau = 4 * dt_crit

# Jacobian template
_jac_slot_placeholder = 1e-300
A_D_jac_template = A_D.tolil()
A_D_jac_template[slip_idx, 0] = _jac_slot_placeholder
A_D_jac_template[slip_idx, vtt_idx] = _jac_slot_placeholder
A_D_jac_template = A_D_jac_template.tocsr()

def rhs_jac_aug(t, y):
    J = A_D_jac_template.copy()
    v2 = y[vtt_idx]
    J[slip_idx, vtt_idx] = v2 / np.sqrt(v2**2 + eps_slip)
    return J

# ── NCP contact backend ────────────────────────────────────────────────
from solve_nivp.ncp_contact import build_ncp_contact

def contact_s0_force_vec(y):
    return np.array([s11_eff_0])

def contact_w0_force_vec(y, k):
    nudge = 1.001
    return np.array([-s11_eff_0 * mu_fric(0.0, 0.0) * nudge])

cs_ncp = build_ncp_contact(
    A=E_D,
    rhs_smooth=rhs_plant,
    y0=y0,
    contacts=contact_blocks,
    C_extract=contact_gap_C,
    D_extract=contact_vel_C,
    B=contact_B,
    component_slices=contact_component_slices,
    gap_func=None,
    theta=1.0,
    reaction_units='force',
    get_s0=contact_s0_force_vec,
    get_w0=contact_w0_force_vec,
    rhs_jac=rhs_jac_aug,
    ncp_type='fischer_burmeister',
    normal_r='auto',
    friction_r='auto',
    inactive_handling='ncp',
)

n_aug_total = cs_ncp.y0.size
nl_atol_ncp = np.full(n_aug_total, 1e-8)
nl_rtol_ncp = np.full(n_aug_total, 1e-6)
nl_atol_ncp[ndofs_aug:] = 1e-10
nl_rtol_ncp[ndofs_aug:] = 0.0

solver_opts_ncp = dict(cs_ncp.solver_opts)
solver_opts_ncp.pop('cold_start_slices', None)
solver_opts_ncp['damped_step_fraction'] = 1.0
solver_opts_ncp['diagonal_regularization'] = 0.0
solver_opts_ncp.update(
    tol=1e-6,
    max_iter=500,
    rhs_jac=cs_ncp.rhs_jac,
    linear_solver='petsc',
    globalization='none',
)

integrator_opts_ncp = dict(cs_ncp.integrator_opts)
if METHOD == 'radau_iia':
    _use_cn = bool(int(__import__('os').environ.get('RADAU_CN', '1')))
    integrator_opts_ncp.update(stages=2, use_coupled_newton=_use_cn)

solve_kwargs = dict(
    fun=cs_ncp.rhs,
    t_span=t_span,
    y0=cs_ncp.y0,
    A=cs_ncp.A,
    method=METHOD,
    projection=cs_ncp.projection,
    solver='semismooth_newton',
    solver_opts=solver_opts_ncp,
    nl_atol=nl_atol_ncp,
    nl_rtol=nl_rtol_ncp,
    component_slices=cs_ncp.component_slices,
    integrator_opts=integrator_opts_ncp,
    h0=h0_init,
    dae_var_weight='auto',
    active_set_filter=False,
    verbose=True,
)

if USE_ADAPTIVE:
    solve_kwargs.update(
        adaptive=True,
        rtol=5e-3,
        atol=1e-4,
        skip_error_indices=[0, len(cs_ncp.component_slices) - 1],
    )
else:
    solve_kwargs.update(adaptive=False)

print(f"\n{'='*64}")
print(f"Running Radau+NCP: h0={h0_init:.4g}, n_steps={n_steps_init}, tmax={tmax}")
print(f"ηs={ηs}, L={L}")
print(f"n_aug_total={n_aug_total}")

# Print auto-r diagnostic info
print(f"\n  E_D diagonal at vtn_idx={vtn_idx}: {E_D[vtn_idx, vtn_idx]:.6e}")
print(f"  E_D diagonal at vtt_idx={vtt_idx}: {E_D[vtt_idx, vtt_idx]:.6e}")
A_diag = np.abs(np.asarray(E_D.diagonal()).ravel())
print(f"  E_D diagonal range: [{A_diag[A_diag>0].min():.3e}, {A_diag.max():.3e}]")
B_dense = contact_B.toarray()
print(f"  contact_B shape: {contact_B.shape}")
print(f"  contact_B nonzeros: {contact_B.nnz}")
r_base_norm = np.sum(B_dense[:, 0]**2 / np.where(A_diag > 0, A_diag, 1.0))
r_base_fric = np.sum(B_dense[:, 1]**2 / np.where(A_diag > 0, A_diag, 1.0))
print(f"  r_base_normal = {r_base_norm:.6e}")
print(f"  r_base_friction = {r_base_fric:.6e}")
print(f"  auto normal_r (h²*base) = {h0_init**2 * r_base_norm:.6e}")
print(f"  auto friction_r (h*base) = {h0_init * r_base_fric:.6e}")
print(f"  s11_eff_0 = {s11_eff_0:.6e}")
print(f"  mu(0)*s11_eff_0 = {mu_fric(0,0)*s11_eff_0:.6e}")
print(f"{'='*64}\n")

start = time.time()
t_ncp, y_ncp, h_ncp_hist, fk_ncp, info_ncp = solve_nivp.solve_ivp_ns(**solve_kwargs)
time_ncp = time.time() - start

mode_label = "adaptive" if USE_ADAPTIVE else "fixed-step"
n_steps_taken = len(t_ncp) - 1
h_arr = np.asarray(h_ncp_hist, dtype=float) if h_ncp_hist is not None else np.array([])
print(f"\nRadau+NCP ({mode_label}) elapsed: {time_ncp:.2f} s")
print(f"  h0              : {h0_init:.4g} s")
print(f"  steps taken     : {n_steps_taken}")
if h_arr.size:
    print(f"  step range      : [{float(h_arr.min()):.3e}, {float(h_arr.max()):.3e}] s")
    print(f"  median step     : {float(np.median(h_arr)):.3e} s")
print(f"  reached t = {float(t_ncp[-1]):.6g}")

# State evolution at key time points
print("\n=== State evolution at right boundary ===")
print(f"  {'step':>5s} {'t':>8s} {'v2':>14s} {'u2':>14s} {'v1':>14s} {'p':>14s} {'f_el_v2':>12s} {'f_el_v1':>12s}")
sample_steps = list(range(0, min(n_steps_taken+1, 50), 10))
sample_steps += list(range(50, min(n_steps_taken+1, 140), 20))
sample_steps += list(range(140, min(n_steps_taken+1, 200), 2))  # dense near instability
sample_steps += list(range(200, n_steps_taken+1, 20))
if n_steps_taken not in sample_steps:
    sample_steps.append(n_steps_taken)
sample_steps = sorted(set(sample_steps))
for si in sample_steps:
    yy = y_ncp[si]
    v2_val = yy[vtt_idx]
    u2_val = yy[u2t_idx]
    slip_val = yy[slip_idx]
    mu_val = mu_fric(slip_val, 0)
    r_N = yy[ndofs_aug] if ndofs_aug < len(yy) else 0
    r_T = yy[ndofs_aug+1] if ndofs_aug+1 < len(yy) else 0
    # Compute RHS force and trace all fields at right boundary
    rhs_force = A_D @ yy[:ndofs_aug]
    f_elastic_v2 = rhs_force[vtt_idx]
    _, _, vtn_r, vtt_r, u1n_r, u2t_r, pt_r = TPloc_idx["right"]
    v1_val = yy[vtn_r]; r1_val = yy[TPloc_idx["right"][0]]  # v1, using same idx
    p_val = yy[pt_r]
    f_elastic_v1 = rhs_force[vtn_r]
    print(f"  {si:5d} {t_ncp[si]:8.4f} {v2_val:14.6e} {u2_val:14.6e} {v1_val:14.6e} {p_val:14.6e} {f_elastic_v2:12.4e} {f_elastic_v1:12.4e}")

# Check for NaN/Inf
if np.any(~np.isfinite(y_ncp)):
    print("\n*** WARNING: NaN/Inf found in solution! ***")
    nan_steps = np.where(~np.all(np.isfinite(y_ncp), axis=1))[0]
    print(f"  First NaN/Inf at step index: {nan_steps[0] if len(nan_steps) else 'none'}")
    if len(nan_steps):
        print(f"  t at that step: {t_ncp[nan_steps[0]]:.6g}")

# Check convergence info
if isinstance(info_ncp, (list, tuple)) and len(info_ncp) > 0:
    if isinstance(info_ncp[0], tuple):
        converged = [s[1] for s in info_ncp]
        iters = [s[2] for s in info_ncp]
        n_failed = sum(1 for c in converged if not c)
        print(f"\n  Convergence stats:")
        print(f"    failed steps: {n_failed}")
        print(f"    max iters: {max(iters)}")
        print(f"    mean iters: {np.mean(iters):.1f}")
        if n_failed > 0:
            fail_idx = [i for i, c in enumerate(converged) if not c]
            fi = fail_idx[0]
            print(f"    first failure at step {fi}, t≈{t_ncp[min(fi+1, len(t_ncp)-1)]:.6g}")
            print(f"    iters at failure: {iters[fi]}")

# Analyze state near failure
print("\n=== State near failure ===")
fail_step = n_steps_taken  # last successful step
y_fail = y_ncp[fail_step][:ndofs_aug]
slip_fail = y_fail[slip_idx] if slip_idx < len(y_fail) else 0.0
v2_fail = y_fail[vtt_idx] if vtt_idx < len(y_fail) else 0.0
print(f"  t_fail = {t_ncp[fail_step]:.6g}")
print(f"  slip = {slip_fail:.6e}")
print(f"  v2(right) = {v2_fail:.6e}")
print(f"  mu_fric(slip, 0) = {mu_fric(slip_fail, 0):.6f}")
print(f"  Dc = {Dc:.6e}")
print(f"  slip/Dc = {slip_fail/Dc:.3f}")

# Check the full augmented state for NaN/Inf
y_full_fail = y_ncp[fail_step]
print(f"  ||y||_inf = {np.max(np.abs(y_full_fail)):.6e}")
print(f"  reaction block range: [{y_full_fail[ndofs_aug:].min():.6e}, {y_full_fail[ndofs_aug:].max():.6e}]")

# Newton iteration history near failure
print("\n=== Iteration history near failure ===")
for j in range(max(0, fail_step-5), min(len(iters), fail_step+2)):
    print(f"  step {j}: t={t_ncp[min(j+1,len(t_ncp)-1)]:.5f}, iters={iters[j]}, ok={converged[j]}")

# Interior field diagnostics during locked phase
print("\n=== Interior field diagnostics ===")
split_idx = basis.split_indices()
v_idx, r_idx, u_idx, p_idx = split_idx
# Reconstruct sub-indices for v2, u2, p
vb_split = basis.split_bases()[0].split_indices()
v2_dof_idx = v_idx[vb_split[1]]  # all v2 DOFs
ub_split = basis.split_bases()[2].split_indices()
u2_dof_idx = u_idx[ub_split[1]]  # all u2 DOFs
p_dof_idx = p_idx  # all pressure DOFs

# Get nodal x coordinates for v2 (P2), u2 (P2), p (P1)
x_mesh = mesh.p[0]  # node x coords
x_all = np.sort(np.unique(np.concatenate([mesh.p[0], 0.5*(mesh.p[0, mesh.t[0]] + mesh.p[0, mesh.t[1]])])))

diag_steps = [0, 50, 100, 130, 150, 170, 200, 250]
diag_steps = [s for s in diag_steps if s <= n_steps_taken]
if n_steps_taken not in diag_steps:
    diag_steps.append(n_steps_taken)

for si in diag_steps:
    yy = y_ncp[si][:ndofs_aug]
    v2_field = yy[v2_dof_idx]
    u2_field = yy[u2_dof_idx]
    p_field = yy[p_dof_idx]
    print(f"\n  step {si}, t={t_ncp[si]:.4f}:")
    print(f"    v2 range: [{v2_field.min():.6e}, {v2_field.max():.6e}]")
    print(f"    u2 range: [{u2_field.min():.6e}, {u2_field.max():.6e}]")
    print(f"    p  range: [{p_field.min():.6e}, {p_field.max():.6e}]")
    print(f"    ||v2||_2 = {np.linalg.norm(v2_field):.6e}")
    print(f"    ||p||_2  = {np.linalg.norm(p_field):.6e}")
    # Separate elastic contribution: A_bulk (no Robin) vs A_robin
    rhs_bulk = Abulk @ yy[:ndofs]
    rhs_robin_l = Abndr_left @ yy[:ndofs]
    rhs_robin_r = Abndr_right @ yy[:ndofs]
    f_bulk_v2 = rhs_bulk[vtt_idx]
    f_robin_l_v2 = rhs_robin_l[vtt_idx]
    f_robin_r_v2 = rhs_robin_r[vtt_idx]
    print(f"    f_bulk_v2  = {f_bulk_v2:.6e} (interior elastic+viscous+Biot)")
    print(f"    f_robin_L  = {f_robin_l_v2:.6e} (left Robin)")
    print(f"    f_robin_R  = {f_robin_r_v2:.6e} (right Robin)")
    print(f"    f_total    = {f_bulk_v2 + f_robin_l_v2 + f_robin_r_v2:.6e}")
