"""Test injection-induced friction change: verify asymmetric response to ±Q0.

Source placed near the fault (right boundary) so the pressure front arrives
within the simulation window. Two runs: +Q0 (expansion, should weaken friction)
and -Q0 (compression, should strengthen). Compares final slip."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import block_diag as sp_block_diag, vstack as sp_vstack
from scipy.sparse.linalg import splu

from skfem import (Basis, ElementVector, ElementComposite, FacetBasis,
                   ElementLineP2, ElementLineP1, MeshLine)
from skfem.assembly import asm, BilinearForm, LinearForm
from skfem.helpers import grad, dot, ddot

import autograd.numpy as anp
from autograd import elementwise_grad

import solve_nivp
from solve_nivp.ncp_contact import build_ncp_contact
import time as timer

# ── Physical parameters ──────────────────────────────────────────────────
L = 1e-3
D = 1000.0           # domain [m]

ρs = 2500 * L**3
ρf = 1000 * L**3
Kf = 2.2e9 * L
Ks = 80e9 * L
nu = 0.15
φ = 0.2
α = 0.6
turt = 1.0

k_perm = 1e-18 * L**(-2)
μf = 1e-3 * L
ηs = 9e-1

Kb = Ks * (1 - α)
G = 1.5 * Kb * (1 - 2*nu) / (1 + nu)
M = ((α - φ)/Ks + φ/Kf)**(-1)
mu_lame = G
lam = Kb - 2*G/3
b_darcy = (k_perm / μf)**(-1)

ρ11 = (1 - φ)*ρs + (turt - 1)*φ*ρf
ρ22 = turt * φ * ρf
ρ12 = -(turt - 1) * φ * ρf

k_vv = 1e18 * L**3
k_vu = 1e18 * L**3
k_r  = 1e21 * L**3

robin_left  = (k_vv, k_vv, k_vu, k_vu, k_r)
robin_right = (k_vv, 0, k_vu, 0, k_r)

Dc = 0.10 / 1000 / L
Dmu = -0.1
mu_res = 0.5
s11_eff_0 = 4e6 * L

Ku = Kb + α**2 * M
rho_b = (1 - φ)*ρs + φ*ρf
vw = np.sqrt((Ku + 4/3*mu_lame) / rho_b)

# ── Mesh ──────────────────────────────────────────────────────────────────
mesh_elements = 41
xmin_d, xmax_d = -D/(2*L), D/(2*L)
xcoords = np.linspace(xmin_d, xmax_d, mesh_elements + 1)
bnd_labels = {
    'left':  lambda x: np.isclose(x[0], xmin_d),
    'right': lambda x: np.isclose(x[0], xmax_d),
}
mesh = MeshLine(xcoords).with_boundaries(bnd_labels)
left_facets  = mesh.boundaries['left']
right_facets = mesh.boundaries['right']

# ── FE spaces ─────────────────────────────────────────────────────────────
dim = 2
intorder = 4
el_v = ElementVector(ElementLineP2(), dim=dim)
el_r = ElementVector(ElementLineP2(), dim=dim)
el_u = ElementVector(ElementLineP2(), dim=dim)
el_p = ElementLineP1()

mixed_element = ElementComposite(el_v, el_r, el_u, el_p)
basis = Basis(mesh, mixed_element, intorder=intorder)
fbasis_left  = FacetBasis(mesh, mixed_element, intorder=intorder, facets=left_facets)
fbasis_right = FacetBasis(mesh, mixed_element, intorder=intorder, facets=right_facets)

def make_sym(gradu):
    o = np.zeros((2, 2, *gradu.shape[2:]), dtype=gradu.dtype)
    o[0, 0] = gradu[0, 0]
    o[0, 1] = 0.5 * gradu[1, 0]
    o[1, 0] = 0.5 * gradu[1, 0]
    return o

def my_sym_grad(u):
    return make_sym(grad(u))

def sij_core(E_kl, x_type):
    d = E_kl.shape[-1]
    if d == 2:
        bs = E_kl.shape[:-2]
        E3 = np.zeros(bs + (3, 3), dtype=E_kl.dtype)
        E3[..., :d, :d] = E_kl
    else:
        E3 = E_kl
    trE = np.trace(E3, axis1=-2, axis2=-1)[..., None, None]
    I3 = np.eye(3, dtype=E3.dtype)
    s3 = 2.0 * mu_lame * E3 + lam * trE * I3
    return s3[..., :d, :d]

def Sij(E, x_type):
    E = np.asarray(E)
    E_last = np.moveaxis(E, (0, 1), (-2, -1))
    S_last = sij_core(E_last, x_type)
    return np.moveaxis(S_last, (-2, -1), (0, 1))

# ── Bilinear forms ────────────────────────────────────────────────────────
@BilinearForm
def varform_lhs(dv, dr, du, dp, δv, δr, δu, δp, w):
    return (ρ11*dot(dv, δv) + ρ22*dot(dr, δr)
            + ρ12*dot(dv, δr) + ρ12*dot(dr, δv)
            + dot(du, δu) + dp*δp)

@BilinearForm
def varform_rhs(v, r, u, p, δv, δr, δu, δp, w):
    Eps = my_sym_grad(u)
    Su = Sij(Eps, "quad")
    dEps = my_sym_grad(v)
    Sv = ηs * Sij(dEps, "quad")
    I = np.eye(Eps.shape[0])[:, :, None, None] * np.ones_like(Eps)
    S = Su + Sv - α * I * p
    t1 = ddot(S, my_sym_grad(δv)) + b_darcy*dot(r, δr) - dot(v, δu)
    t2 = -dot(p, grad(δr)[0, 0]) + M*dot(α*grad(v)[0, 0] + grad(r)[0, 0], δp)
    return t1 + t2

@BilinearForm
def varform_robin(v, r, u, p, δv, δr, δu, δp, w):
    c_vv1, c_vv2, c_vu1, c_vu2, c_r = w.params
    rn = r[0]
    δrn = δr[0]
    t1 = (c_vv1*v[0] + c_vu1*u[0])*δv[0]
    t1 += (c_vv2*v[1] + c_vu2*u[1])*δv[1]
    t2 = c_r * rn * δrn
    return t1 + t2

@BilinearForm
def varform_effective_traction(v, r, u, p, δv, δr, δu, δp, w):
    return α * p * dot(w.n, δv)

# ── Assembly ──────────────────────────────────────────────────────────────
E_mat = asm(varform_lhs, basis)
Abulk = -asm(varform_rhs, basis)
Abndr_left  = -asm(varform_robin, fbasis_left,  params=robin_left)
Abndr_right = -asm(varform_robin, fbasis_right, params=robin_right)
A_mat = Abulk + Abndr_left + Abndr_right

A_eff_right = -asm(varform_effective_traction, fbasis_right)
A_mat = A_mat - A_eff_right

ndofs = basis.N

# ── Source near fault ─────────────────────────────────────────────────────
source_frac = 0.95
point_source_loc_nd = (D / 2) * source_frac / L

dx = (np.amax(mesh.p) - np.amin(mesh.p)) / mesh_elements
sources_RMS_width = 1.5 * dx

Bp_full = lil_matrix((ndofs, 1))
sigma_src = sources_RMS_width
@LinearForm
def smooth_dirac(δv, δr, δu, δp, w):
    r2 = (w.x[0] - point_source_loc_nd)**2
    g = np.exp(-r2 / (2 * sigma_src**2))
    norm_factor = sigma_src * np.sqrt(2 * np.pi)
    return g / norm_factor * δp / dx

fvec = asm(smooth_dirac, basis)
Bp_full[:, 0] = fvec.reshape(-1, 1) / L
B_source = M * Bp_full.tocsr()

# ── Dirichlet BCs ─────────────────────────────────────────────────────────
user_to_skfem = {"v1": "u^1^1", "v2": "u^2^1", "u1": "u^1^3", "u2": "u^2^3", "p": "u^4"}
bc_dofs = {'u1': {'left': 0.}, 'u2': {'left': 0.}}

E_D = E_mat.tolil(); A_D = A_mat.tolil(); B_D = B_source.tolil()
dofs_D = np.array([], dtype=np.int32)
for field, sides in bc_dofs.items():
    for side, val in sides.items():
        dofs_D = np.append(dofs_D, basis.get_dofs(side).all(user_to_skfem[field]))
E_D[dofs_D, :] = 0.; E_D[:, dofs_D] = 0.; E_D[dofs_D, dofs_D] = 1.
A_D[dofs_D, :] = 0.; A_D[:, dofs_D] = 0.
B_D[dofs_D] = 0.
E_D = E_D.tocsr(); A_D = A_D.tocsr(); B_D = B_D.tocsr()

# ── Augment with slip DOF ─────────────────────────────────────────────────
slip_idx = ndofs
eps_slip = 1e-20 / L
slip_rate_floor = np.sqrt(eps_slip)
E_D = sp_block_diag([E_D, csr_matrix(np.array([[1.]]))], format='csr')
A_D = sp_block_diag([A_D, csr_matrix(np.array([[0.]]))], format='csr')
B_D = sp_vstack([B_D, csr_matrix((1, B_D.shape[1]))], format='csr')
ndofs_aug = ndofs + 1

# ── Contact DOF indices ───────────────────────────────────────────────────
sides = ["right"]
TPloc_idx = {}
for side in sides:
    t1  = basis.get_dofs(side).all(user_to_skfem["v1"])[0]
    t2  = basis.get_dofs(side).all(user_to_skfem["v2"])[0]
    u1n = basis.get_dofs(side).all(user_to_skfem["u1"])[0]
    u2t = basis.get_dofs(side).all(user_to_skfem["u2"])[0]
    p_n = basis.get_dofs(side).all(user_to_skfem["p"])[0]
    TPloc_idx[side] = [t1, t2, t1, t2, u1n, u2t, p_n]

_, _, vtn_idx, vtt_idx, u1n_idx, u2t_idx, pt_idx = TPloc_idx["right"]

# ── Friction ──────────────────────────────────────────────────────────────
def mu_fric(slip, slip_rate):
    return mu_res * (1.0 - Dmu/mu_res * np.exp(-slip / Dc))

def mu_(z):
    return mu_fric(z[slip_idx], z[vtt_idx])

# ── Smoothed source pulse ─────────────────────────────────────────────────
def smoothstep_quintic(x):
    return x*x*x*(x*(x*6 - 15) + 10)

def smooth_step(t, t0, tau):
    x = anp.clip((t - t0) / tau, 0.0, 1.0)
    return smoothstep_quintic(x)

def smooth_pulse(t, t_start, t_end, tau=1e-4):
    return (1 - smooth_step(t, t_end, tau)) * smooth_step(t, t_start, tau)

dsmooth_pulse = elementwise_grad(smooth_pulse, argnum=0)

# ── Solver ────────────────────────────────────────────────────────────────
def run_case(Q0_val, nudge, tmax, n_steps, label=""):
    point_source_Q0 = np.array([[Q0_val]])

    dx_mesh = (np.amax(mesh.p) - np.amin(mesh.p)) / mesh_elements
    dt_crit = dx_mesh / vw
    tau_smooth = 4 * dt_crit

    def qv(t, t_start, t_end, tau=tau_smooth):
        return point_source_Q0[:] * smooth_pulse(t, t_start, t_end, tau) / L

    def dqv(t, t_start, t_end, tau=tau_smooth):
        return point_source_Q0[:] * dsmooth_pulse(t, t_start, t_end, tau) / L

    dq = lambda t: dqv(t, t_start=0, t_end=5*tmax/3, tau=tau_smooth).reshape(-1)

    def rhs_plant(t, y):
        rhs_sm = A_D @ y
        rhs_sm = rhs_sm + B_D @ dq(t)
        v2 = y[vtt_idx]
        rhs_sm[slip_idx] = np.sqrt(v2**2 + eps_slip) - slip_rate_floor
        return rhs_sm

    _jac_placeholder = 1e-300
    A_D_jac_tmpl = A_D.tolil()
    A_D_jac_tmpl[slip_idx, 0] = _jac_placeholder
    A_D_jac_tmpl[slip_idx, vtt_idx] = _jac_placeholder
    A_D_jac_tmpl = A_D_jac_tmpl.tocsr()

    def rhs_jac_aug(t, y):
        J = A_D_jac_tmpl.copy()
        v2 = y[vtt_idx]
        J[slip_idx, vtt_idx] = v2 / np.sqrt(v2**2 + eps_slip)
        return J

    y0 = np.zeros(ndofs_aug)

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
        sorted(set(range(ndofs_aug)) - set(contact_vel_idx.tolist())), dtype=int)
    contact_component_slices = [contact_vel_idx, contact_other_idx]

    def contact_s0(y):
        return np.array([s11_eff_0])

    def contact_w0(y, k):
        return np.array([-s11_eff_0 * mu_fric(0.0, 0.0) * nudge])

    cs = build_ncp_contact(
        A=E_D, rhs_smooth=rhs_plant, y0=y0,
        contacts=contact_blocks, C_extract=contact_gap_C, D_extract=contact_vel_C,
        B=contact_B, component_slices=contact_component_slices,
        gap_func=None, theta=1.0, reaction_units='force',
        get_s0=contact_s0, get_w0=contact_w0,
        rhs_jac=rhs_jac_aug,
        ncp_type='fischer_burmeister', normal_r='auto', friction_r='auto',
        inactive_handling='ncp',
    )

    n_total = cs.y0.size
    nl_atol = np.full(n_total, 1e-8)
    nl_rtol = np.full(n_total, 1e-6)
    nl_atol[ndofs_aug:] = 1e-10
    nl_rtol[ndofs_aug:] = 0.0

    sopts = dict(cs.solver_opts)
    sopts.pop('cold_start_slices', None)
    sopts['damped_step_fraction'] = 1.0
    sopts['diagonal_regularization'] = 0.0
    sopts.update(tol=1e-8, max_iter=500, rhs_jac=cs.rhs_jac, linear_solver='splu')

    iopts = dict(cs.integrator_opts)
    iopts.update(stages=2, use_coupled_newton=True)

    h0 = tmax / n_steps
    t0 = timer.time()
    t_arr, y_arr, h_hist, fk, info = solve_nivp.solve_ivp_ns(
        fun=cs.rhs, t_span=(0.0, tmax), y0=cs.y0, A=cs.A,
        method='radau_iia', projection=cs.projection,
        solver='semismooth_newton', solver_opts=sopts,
        nl_atol=nl_atol, nl_rtol=nl_rtol,
        component_slices=cs.component_slices,
        integrator_opts=iopts, h0=h0,
        dae_var_weight='auto', active_set_filter=False, adaptive=False,
    )
    elapsed = timer.time() - t0

    slip_hist = y_arr[:, slip_idx]
    r_hist = y_arr[:, ndofs_aug:]
    Rn = r_hist[:, 0]
    Rt = r_hist[:, 1] if r_hist.shape[1] >= 2 else np.zeros_like(t_arr)

    p_vals = np.array([y_arr[i, pt_idx] for i in range(len(t_arr))]) / L
    v2_vals = np.array([y_arr[i, vtt_idx] for i in range(len(t_arr))]) * L

    n_ok = sum(1 for _, ok, _ in info if ok)
    print(f"  [{label}] Q0={Q0_val:+.1e}, elapsed={elapsed:.1f}s, "
          f"steps={n_ok}/{len(info)}, "
          f"final_slip={slip_hist[-1]*L:.4e} m, "
          f"R_n range=[{Rn.min():.3e},{Rn.max():.3e}], "
          f"p(x_R) range=[{p_vals.min():.3e},{p_vals.max():.3e}] Pa")

    # Extract σ'₁₁ and p at several x-locations
    probe_fracs = [0.0, 0.5, 0.90, 0.95, 0.99]
    probe_x = [(D/2) * f / L for f in probe_fracs]
    probe_labels = [f'x/x_R={f:.2f}' for f in probe_fracs]
    u_dofs_all = basis.get_dofs().all('u^1^3')
    p_dofs_all = basis.get_dofs().all('u^4')
    x_dof_u = basis.doflocs[0, u_dofs_all]
    x_dof_p = basis.doflocs[0, p_dofs_all]

    sig_eff_probes = np.zeros((len(t_arr), len(probe_x)))
    p_probes = np.zeros((len(t_arr), len(probe_x)))
    for ip, xp in enumerate(probe_x):
        idx_u = np.argmin(np.abs(x_dof_u - xp))
        idx_p = np.argmin(np.abs(x_dof_p - xp))
        dof_u = u_dofs_all[idx_u]
        dof_p = p_dofs_all[idx_p]
        for it in range(len(t_arr)):
            u_val = y_arr[it, dof_u]
            p_val = y_arr[it, dof_p]
            p_probes[it, ip] = p_val / L
            # Approximate du/dx from FD between adjacent DOFs
            if idx_u < len(u_dofs_all) - 1:
                dof_u_next = u_dofs_all[idx_u + 1]
                dx_u = x_dof_u[idx_u + 1] - x_dof_u[idx_u]
                eps_11 = (y_arr[it, dof_u_next] - y_arr[it, dof_u]) / dx_u
            else:
                dof_u_prev = u_dofs_all[idx_u - 1]
                dx_u = x_dof_u[idx_u] - x_dof_u[idx_u - 1]
                eps_11 = (y_arr[it, dof_u] - y_arr[it, dof_u_prev]) / dx_u
            sig_eff_probes[it, ip] = (2*mu_lame + lam) * eps_11 / L

    return dict(t=t_arr, slip=slip_hist, Rn=Rn, Rt=Rt, p_right=p_vals,
                v2_right=v2_vals, r_hist=r_hist, y=y_arr, label=label,
                sig_eff_probes=sig_eff_probes, p_probes=p_probes,
                probe_labels=probe_labels)


# ── Run experiments ───────────────────────────────────────────────────────
print(f"Source at x = {source_frac*D/2:.0f} m (fault at x = {D/2:.0f} m)")
print(f"s0 = {s11_eff_0:.1f} FE = {s11_eff_0/L:.1e} Pa")
print(f"μ(0) = {mu_fric(0,0):.2f}, μ_res = {mu_res:.2f}")
print(f"Diffusion length at t=5: {np.sqrt(k_perm/μf * M * 5)*L*1e3:.1f} mm")
print(f"Wave travel to fault: {(D/2*(1-source_frac))/vw/L:.4f} s")
print()

tmax = 5.0
n_steps = 1000
nudge = 0.999
Q0_mag = 1e-2

print(f"nudge = {nudge} (margin = {(1-nudge)*100:.1f}% of μ·s0)")
print(f"|Q0| = {Q0_mag:.1e}")
print()

res_pos = run_case(+Q0_mag, nudge, tmax, n_steps, label="+Q0 (expansion)")
res_neg = run_case(-Q0_mag, nudge, tmax, n_steps, label="-Q0 (compression)")
res_none = run_case(0.0,    nudge, tmax, n_steps, label="Q0=0 (baseline)")

# ── Compare ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)
slip_pos  = res_pos['slip'][-1] * L
slip_neg  = res_neg['slip'][-1] * L
slip_none = res_none['slip'][-1] * L
print(f"  Final slip (+Q0, expansion):   {slip_pos:.6e} m")
print(f"  Final slip (-Q0, compression): {slip_neg:.6e} m")
print(f"  Final slip (baseline):         {slip_none:.6e} m")
print()

if slip_pos > slip_none and slip_neg < slip_none:
    print("  ✓ CORRECT: +Q0 (weaker friction) → more slip, -Q0 (stronger) → less slip")
elif slip_pos > slip_neg:
    print("  ~ PARTIAL: +Q0 gives more slip than -Q0, but baseline comparison unclear")
else:
    print("  ✗ UNEXPECTED: -Q0 gives more or equal slip — coupling may be wrong")

print(f"\n  Slip ratio (+Q0 / baseline): {slip_pos/slip_none:.4f}")
print(f"  Slip ratio (-Q0 / baseline): {slip_neg/slip_none:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

ax = axes[0]
for res in [res_pos, res_neg, res_none]:
    ax.plot(res['t'], res['slip'] * L, lw=2, label=res['label'])
ax.set_ylabel('slip [m]')
ax.set_title('Accumulated slip')
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[1]
for res in [res_pos, res_neg, res_none]:
    Rn_s0_Pa = (res['Rn'] + s11_eff_0) / L
    ax.plot(res['t'], Rn_s0_Pa, lw=1.5, label=res['label'])
ax.axhline(s11_eff_0 / L, color='k', lw=0.8, ls=':', label=f's0 = {s11_eff_0/L:.1e} Pa')
ax.set_ylabel('$R_n + s_0$ [Pa]')
ax.set_title('Contact normal force (friction capacity driver)')
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[2]
for res in [res_pos, res_neg, res_none]:
    ax.plot(res['t'], res['p_right'], lw=1.5, label=res['label'])
ax.set_ylabel('$p(x_R)$ [Pa]')
ax.set_title('Pore pressure at fault')
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[3]
for res in [res_pos, res_neg, res_none]:
    mu_s = np.array([mu_fric(float(s), 0) for s in res['slip']])
    w0 = -s11_eff_0 * mu_fric(0, 0) * nudge
    tau = np.abs(w0 + res['Rt'])
    cap = mu_s * (res['Rn'] + s11_eff_0)
    margin = cap - tau
    ax.plot(res['t'], margin, lw=1.5, label=res['label'])
ax.axhline(0.0, color='k', lw=0.8)
ax.set_ylabel('friction margin (FE units)')
ax.set_title('$\\mu(R_n+s_0) - |R_t+w_0|$')
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[4]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(res_pos['probe_labels'])))
for ip, (lab, c) in enumerate(zip(res_pos['probe_labels'], colors)):
    ax.plot(res_pos['t'], res_pos['sig_eff_probes'][:, ip], lw=1.5, ls='-', color=c,
            label=f'+Q0 {lab}')
    ax.plot(res_neg['t'], res_neg['sig_eff_probes'][:, ip], lw=1.5, ls='--', color=c,
            label=f'-Q0 {lab}')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel("$\\sigma'_{11}$ [Pa]")
ax.set_title("Effective stress $\\sigma'_{11}$ (tension-positive: negative = compression)")
ax.legend(fontsize=6, ncol=2)
ax.grid(True)

ax = axes[5]
for ip, (lab, c) in enumerate(zip(res_pos['probe_labels'], colors)):
    ax.plot(res_pos['t'], res_pos['p_probes'][:, ip], lw=1.5, ls='-', color=c,
            label=f'+Q0 {lab}')
    ax.plot(res_neg['t'], res_neg['p_probes'][:, ip], lw=1.5, ls='--', color=c,
            label=f'-Q0 {lab}')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('$p$ [Pa]')
ax.set_title('Pore pressure at probe locations')
ax.set_xlabel('$t$ [s]')
ax.legend(fontsize=6, ncol=2)
ax.grid(True)

plt.tight_layout()
plt.savefig('_test_injection_sign_result.png', dpi=150)
plt.show()
print(f"\nPlot saved to _test_injection_sign_result.png")
