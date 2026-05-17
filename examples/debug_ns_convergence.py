#!/usr/bin/env python
"""
Diagnostic script for sliding_block_demo_ns convergence failure.
Reproduces the notebook setup, then instruments the first BE step
to show exactly what the SSN sees.
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.sparse as sp
from skfem import MeshTri1, MeshQuad1
from skfem.models.elasticity import lame_parameters

# ── Poroelasticity imports ──
sys.path.insert(0, '/home/david/Documents/Poroelasticity')
from poroelasticity.cgporoelastostatics import CGPoroelastostatics
from poroelasticity.mesh_builder import CrackMeshBuilder
from solve_nivp.contact import build_impulse_contact

# ════════════════════════════════════════════════════════════════════
# Parameters (identical to notebook)
# ════════════════════════════════════════════════════════════════════
NU = 0.25; G_SHEAR = 22e3
ETA_FLUID = 2e-10 / 3600; BETA_FLUID = 8.5e-5
K_PERM = 1e-15 * 1e-6; ALPHA_BIOT = 0.0
XMIN, XMAX, YMIN, YMAX = -0.5, 0.5, -0.5, 0.5
SCALE_L = 1.0; SCALE_EPS = 1e-3
CRACK_THETA = np.pi / 2; CRACK_X0 = 0.0; CRACK_Y0 = -0.25
CRACK_LENGTH = 1.0; CRACK_T = 0.0
CRACK_K_N = 0.0; CRACK_K_T = 0.0
CRACK_ETA_N = 0.0; CRACK_ETA_T = 0.0
CRACK_LAW = 'nonsmooth'; ROTATE_TO_NT = True
BULK_MU_V = 1e5; BULK_LAM_V = None
V_PUSH = 1e-5; RHO_G = 26.5
MU_FRICTION = 0.3; E_RESTITUTION = 0.0
N_ELEM = 4; ELEMENT_TYPE = 'quad'; CONFORMING = True
T_END = 100.0; INCLUDE_HESSIAN = False

def body_force_fn(x, t):
    fx = np.zeros_like(x[0])
    fy = -RHO_G * np.ones_like(x[0])
    return np.array([fx, fy])

E_young = 2.0 * G_SHEAR * (1.0 + NU)
lam_e, mu_e = lame_parameters(E_young, NU)
params = (mu_e, lam_e, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)

bc = {
    'v1': {'bottom': 0.0}, 'v2': {'bottom': 0.0},
    'v1_rate': {'top': 0.0}, 'v2_rate': {},
    'dp_rate': {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0},
}
model_params = {
    'T': CRACK_T, 'k_n': CRACK_K_N, 'k_t': CRACK_K_T,
    'eta_n': None, 'eta_t': None,
}

# ════════════════════════════════════════════════════════════════════
# Build mesh & assemble
# ════════════════════════════════════════════════════════════════════
print("Building mesh and assembling system...")
builder = CrackMeshBuilder(
    XMIN, XMAX, YMIN, YMAX, N_ELEM,
    crack_theta=CRACK_THETA, crack_x0=CRACK_X0,
    crack_y0=CRACK_Y0, crack_length=CRACK_LENGTH,
    element_type=ELEMENT_TYPE, conforming=CONFORMING,
)
mesh, el_p, el_u, crack, h = builder.build()

poro = CGPoroelastostatics(
    mesh=mesh, element_p=el_p, element_u=el_u,
    params=params, crack=crack, intorder=10,
    model_params=model_params, scales=(SCALE_L, SCALE_EPS),
    bc=bc, P_scale=1.0, verbose=False, free_memory=True,
    enforcement_type='nodal', include_hessian=INCLUDE_HESSIAN,
    include_taylor=True, lumped_coupling='consistent',
    body_force=body_force_fn,
    bulk_viscosity={'mu_v': BULK_MU_V, 'lam_v': BULK_LAM_V},
    crack_law=CRACK_LAW, rotate_crack_to_nt=ROTATE_TO_NT,
)

poro.strip_multiplier_dynamics()
projection, meta = poro.build_projection()

Np = poro.basis_p.N; Nu = poro.basis_u.N
N_orig = poro.ndofs; n_c = poro.n_lambda_q
dim = poro.dim; n_ls = dim * n_c
field_sl = meta['field_sl']; lam_q_sl = meta['lam_q_sl']; lam_s_sl = meta['lam_s_sl']
component_slices = meta['component_slices']

print(f"N_orig={N_orig}, Np={Np}, Nu={Nu}, n_c={n_c}, n_ls={n_ls}")

# ════════════════════════════════════════════════════════════════════
# Build contact system (same as notebook cell 16)
# ════════════════════════════════════════════════════════════════════
C_s_full = meta['C_s_full']
R_nt = meta['R_nt']
n_field = field_sl.stop
n_lam_s = C_s_full.shape[0]

R_block = sp.block_diag([sp.eye(n_ls), R_nt], format='csr')
C_s_nt = R_block @ C_s_full
C_extract = sp.hstack([C_s_nt, sp.csr_matrix((n_lam_s, N_orig - n_field))]).tocsr()

def gap_func(y_phys, t):
    delta_c = np.asarray(C_extract @ y_phys).ravel()
    return delta_c[n_ls:n_ls + n_c]

contacts = [
    dict(vel_normal_idx=n_ls + k, vel_tangential_idx=[n_ls + n_c + k],
         mu=MU_FRICTION, e=E_RESTITUTION)
    for k in range(n_c)
]

jump_start = lam_s_sl.start + n_ls
jump_end = lam_s_sl.stop
B_jump_xy = (-poro.A[:, jump_start:jump_end]).tocsc()
B_jump = (B_jump_xy @ R_nt.T).tocsc()
perm = []
for k in range(n_c):
    perm.append(k); perm.append(n_c + k)
B_contact = B_jump[:, perm].tocsr()

flux_constraint = meta['constraints'][0]
_n_lam_s_dim = lam_s_sl.stop - lam_s_sl.start
zero_lam_s = dict(
    g=lambda zf, *_a, _n=_n_lam_s_dim: np.zeros(_n),
    dg_dy=lambda zf, *_a, _n=_n_lam_s_dim: np.zeros((_n, _n)),
    y_slice=lam_s_sl, q_slice=lam_s_sl,
)

y0_contact = meta['y0'].copy()
y0_contact[lam_s_sl] = 0.0

n_sources = poro.Bp.shape[1] if poro.Bp is not None else 0

def rhs_smooth(t, y):
    q_total = np.full(n_sources, 0.0) if n_sources > 0 else np.array([])
    return poro.rhs(t, y, q_total).squeeze()

_neg_A = (-poro.A).tocsr()
def rhs_jac_smooth(t, y):
    return _neg_A

cs = build_impulse_contact(
    A=poro.M, rhs_smooth=rhs_smooth, rhs_jac=rhs_jac_smooth,
    y0=y0_contact, contacts=contacts, gap_func=gap_func,
    B=B_contact, C_extract=C_extract, D_extract=C_extract,
    rate_form=True, theta=1.0,
    constraints=[flux_constraint, zero_lam_s],
    component_slices=component_slices,
)

N_aug = len(cs.y0)
n_react = N_aug - N_orig

print(f"N_aug={N_aug}, n_react={n_react}")
print(f"cs.projection type = {type(cs.projection).__name__}")

# ════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Simulate one backward Euler step manually
# ════════════════════════════════════════════════════════════════════
T_sc = poro.get_scales()[0]
tmax = T_END / T_sc
h = tmax / 10  # step size

print(f"\n{'='*70}")
print(f"MANUAL BACKWARD EULER STEP: t=0, h={h:.6e}")
print(f"{'='*70}")

y0 = cs.y0.copy()
print(f"\ny0 stats:")
print(f"  ||y0[:N_orig]|| = {np.linalg.norm(y0[:N_orig]):.6e}")
print(f"  ||y0[N_orig:]|| = {np.linalg.norm(y0[N_orig:]):.6e}")
print(f"  y0[:N_orig] nonzero entries: {np.count_nonzero(y0[:N_orig])}")

# ── Evaluate RHS at y0 ──
# The augmented RHS needs (t, y, prev_state, h)
# For the first step, prev_state = y0
f0 = cs.rhs(0.0, y0, y0, h)
print(f"\nRHS at y0:")
print(f"  ||f0[:N_orig]|| = {np.linalg.norm(f0[:N_orig]):.6e}  (physical)")
print(f"  ||f0[N_orig:]|| = {np.linalg.norm(f0[N_orig:]):.6e}  (reactions)")
print(f"  max|f0[:N_orig]| = {np.max(np.abs(f0[:N_orig])):.6e}")
print(f"  max|f0[N_orig:]| = {np.max(np.abs(f0[N_orig:])):.6e}")

# Check what the reaction rows contain
print(f"\n  f0 reaction DOFs (first 10):")
for i in range(min(10, n_react)):
    print(f"    react[{i}] = {f0[N_orig + i]:+.6e}")

# ── Evaluate Jacobian at y0 ──
J0 = cs.rhs_jac(0.0, y0, y0, h)
print(f"\nJacobian at y0:")
if sp.issparse(J0):
    print(f"  Shape: {J0.shape}, nnz={J0.nnz}")
    print(f"  ||J0||_F = {sp.linalg.norm(J0):.6e}")
    J0d = J0.toarray()
else:
    print(f"  Shape: {J0.shape} (dense)")
    print(f"  ||J0||_F = {np.linalg.norm(J0):.6e}")
    J0d = np.array(J0)

# Check diagonal
diag_J = np.diag(J0d)
zero_diag = np.where(diag_J == 0)[0]
print(f"  Zero diags: {len(zero_diag)} at indices: {zero_diag[:20]}...")
print(f"  Diag range: [{diag_J.min():.6e}, {diag_J.max():.6e}]")

# ── Check what the SSN residual F would be ──
# BE residual: F(y) = A y - A y0 - h * f(t+h, y, y0, h)
A_aug = cs.A
if sp.issparse(A_aug):
    A_d = A_aug.toarray()
else:
    A_d = np.array(A_aug)

# At y=y0 (first iteration): F = A*y0 - A*y0 - h*f(h, y0, y0, h)
#                              = -h * f(h, y0, y0, h)
f_at_h = cs.rhs(h, y0, y0, h)
F0 = -h * f_at_h
print(f"\nBE residual F(y0) = -h * f(h, y0, y0, h):")
print(f"  ||F0|| = {np.linalg.norm(F0):.6e}")
print(f"  ||F0[:N_orig]|| = {np.linalg.norm(F0[:N_orig]):.6e}")
print(f"  ||F0[N_orig:]|| = {np.linalg.norm(F0[N_orig:]):.6e}")

# ── Check the contact projection at y0 ──
# candidate = y0 - lam * J_res \ F
# But first, what does the projection of y0 look like?
proj = cs.projection
print(f"\nProjection at y0:")
y0_proj = proj.project(y0, y0)
print(f"  y0_proj == y0? {np.allclose(y0_proj, y0)}")
print(f"  ||y0 - y0_proj|| = {np.linalg.norm(y0 - y0_proj):.6e}")

# ── Check gap function ──
gap = gap_func(y0[:N_orig], 0.0)
print(f"\nGap at y0:")
print(f"  gap values: {gap}")
print(f"  min gap = {gap.min():.6e}  (negative = contact active)")
print(f"  max gap = {gap.max():.6e}")

# ── Check what C_extract @ y0 gives ──
Cy0 = np.asarray(C_extract @ y0[:N_orig]).ravel()
print(f"\nC_extract @ y0:")
print(f"  Average block (should be ~0): ||Cy0[:n_ls]|| = {np.linalg.norm(Cy0[:n_ls]):.6e}")
print(f"  Normal jump [[u_n]]:  {Cy0[n_ls:n_ls+n_c]}")
print(f"  Tangential jump [[u_t]]: {Cy0[n_ls+n_c:2*n_ls]}")

# ── Check A_aug structure ──
print(f"\nA_aug (mass matrix) structure:")
A_diag = A_d.diagonal()
print(f"  Shape: {A_aug.shape}")
print(f"  Reaction-block diagonal: {A_diag[N_orig:]}")
print(f"  min phys diag: {A_diag[:N_orig].min():.6e}")
print(f"  max phys diag: {A_diag[:N_orig].max():.6e}")

# ── Eigenvalue analysis of J_res = A \ J ──
# BE Jacobian: J_res = M^{-1} * J  (where J is the rhs Jacobian)
# Actually for SSN: the residual is G(y) = y - y0 - h*M^{-1}*f(h,y,y0,h)
# and J_res = I - h*M^{-1}*J_rhs
# Let's compute what SSN actually sees as J_in
print(f"\n{'='*70}")
print("SSN Jacobian analysis:")
print(f"{'='*70}")

# The integrator forms: G(y) = A(y - y0) - h*f(t+h, y) = 0
# J_G = A - h * J_rhs
# SSN sees J_in from the integrator: J_in = M^{-1} * (A - h*df/dy)
# But actually the integrator wraps it differently...
# Let's check what the actual Jacobian function returns

# J_rhs already includes the time-step structure through the rhs/jac
# Let me check by looking at the integrator

# Actually the SSN function is:
#   F(y) = (y - y_{n-1}) - h * inv(A) * f(t, y, y_{n-1}, h)
# where f is the augmented rhs.
# The residual Jacobian is:
#   J_F = I - h * inv(A) * J_f

# With A = M (mass matrix), J_f = rhs_jac
# J_F = I - h * M^{-1} * J_f

# But wait -- for DAE with singular M, we actually form:
#   F(y) = A*(y - y_{n-1}) - h*f(t, y)
#   J_F = A - h*J_f

print("\nChecking J_F = A - h * J_rhs:")
if sp.issparse(J0):
    J_F = A_aug - h * J0
else:
    J_F = A_d - h * J0d
if sp.issparse(J_F):
    J_Fd = J_F.toarray()
else:
    J_Fd = J_F

# Check rank/conditioning
J_F_diag = np.diag(J_Fd)
print(f"  J_F diag: min={J_F_diag.min():.6e}, max={J_F_diag.max():.6e}")
print(f"  Zero J_F diags: {np.sum(np.abs(J_F_diag) < 1e-30)}")

# For the reaction block specifically:
J_F_react = J_Fd[N_orig:, :]
print(f"  Reaction rows of J_F: shape={J_F_react.shape}")
print(f"  ||reaction rows||_F = {np.linalg.norm(J_F_react):.6e}")
print(f"  Reaction-reaction block diag: {np.diag(J_Fd[N_orig:, N_orig:])}")

# What does the tangent cone D look like?
print(f"\nTangent cone D at y0:")
try:
    D = proj.tangent_cone(y0, y0, rhok=1.0, t=0.0)
    if isinstance(D, tuple):
        Dproj, Dstate = D
        print(f"  D is tuple: Dproj shape={Dproj.shape}, Dstate shape={Dstate.shape if Dstate is not None else None}")
        if sp.issparse(Dproj):
            Dd = Dproj.toarray()
        else:
            Dd = np.array(Dproj)
    else:
        Dd = D.toarray() if sp.issparse(D) else np.array(D)
        Dstate = None
        print(f"  D shape: {Dd.shape}")
    
    D_diag = np.diag(Dd)
    print(f"  D diag range: [{D_diag.min():.6e}, {D_diag.max():.6e}]")
    print(f"  D is identity? {np.allclose(Dd, np.eye(N_aug))}")
    print(f"  D nnz = {np.count_nonzero(Dd)}")
    
    # Check which rows are non-trivial (not identity)
    I_aug = np.eye(N_aug)
    non_id_rows = np.where(np.max(np.abs(Dd - I_aug), axis=1) > 1e-12)[0]
    print(f"  Non-identity rows: {len(non_id_rows)}")
    if len(non_id_rows) > 0 and len(non_id_rows) <= 20:
        for r in non_id_rows:
            print(f"    row {r}: diag={Dd[r,r]:.4e}, off-diag max={np.max(np.abs(Dd[r,:] - I_aug[r,:])):.4e}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ── Check the SSN fixed-point: proj(y - lam * J_F^{-1} * F0) ──
print(f"\n{'='*70}")
print("Manual SSN iteration (first step):")
print(f"{'='*70}")

# Compute J_SSN = I - D + D * (lam * J_in)
# where J_in is what the integrator passes to SSN
# For BackwardEuler: J_in = jacobian of the reduced residual G(y) = y - y0 - h*M^{-1}*f
# Actually, let's look at what BackwardEuler does...

# For now let's try to do what the solver would do:
# 1. Compute residual F = proj(y) - y + lam * J^{-1} @ (A @ (y-y0) - h*f)
#    No wait, the SSN residual is: 
#    F(y) = y - proj_rho(y - rho * G(y)) where G(y) = ... the implicit residual

# Let me just check the sizes and the B matrix more carefully
print(f"\nB_contact structure:")
print(f"  shape: {B_contact.shape}")
print(f"  Physical rows with nonzero B entries: {np.unique(B_contact.nonzero()[0][:20])[:10]}...")
print(f"  B_contact[N_orig-10:N_orig, :5] (last phys rows, first react cols):")
for r in range(max(0, N_orig-5), N_orig):
    row_data = B_contact[r, :min(5, n_react)].toarray().ravel()
    if np.any(row_data != 0):
        print(f"    row {r}: {row_data}")

# Check: which physical rows does B_contact couple to?
B_rows_active = np.unique(B_contact.nonzero()[0])
print(f"  B acts on {len(B_rows_active)} physical rows (of {N_orig})")
print(f"  These are u-DOFs? rows in [{B_rows_active.min()}, {B_rows_active.max()}] vs Np={Np} to Np+Nu={Np+Nu}")

# Check if the reaction rows of J0 are correct
print(f"\nReaction rows of Jacobian J0 (rhs_jac):")
if sp.issparse(J0):
    J0_react = J0[N_orig:, :].toarray()
else:
    J0_react = J0d[N_orig:, :]
print(f"  shape: {J0_react.shape}")
print(f"  ||reaction rows, phys cols|| = {np.linalg.norm(J0_react[:, :N_orig]):.6e}")
print(f"  ||reaction rows, react cols|| = {np.linalg.norm(J0_react[:, N_orig:]):.6e}")
print(f"  reaction-reaction block: {J0_react[:min(6,n_react), N_orig:N_orig+min(6,n_react)]}")
print(f"  reaction-phys (first 5 react × first 5 nonzero phys cols):")
nz_cols = np.where(np.any(J0_react[:, :N_orig] != 0, axis=0))[0][:10]
if len(nz_cols) > 0:
    for ri in range(min(5, n_react)):
        vals = [f"{J0_react[ri, c]:+.4e}" for c in nz_cols[:5]]
        print(f"    react[{ri}]: cols {nz_cols[:5]} = {vals}")

# ── Check what D_extract (velocity extraction) does ──
# With rate_form=True, reaction rows compute:
#   out[rN] = -(v_N + alpha * ||v_T||)
#   where v_N = D_extract[vN_idx, :] @ (yp - yp_prev) / h
# On the FIRST step, if prev_state == y0 and y hasn't changed, v_c = 0
# So all reaction rows should be 0!
print(f"\n{'='*70}")
print("FIRST STEP DIAGNOSIS:")
print(f"{'='*70}")
print(f"With rate_form=True and prev_state=y0:")
print(f"  v_c = D_extract @ (y0 - y0) / h = 0  (zero velocity!)")
print(f"  → All reaction RHS rows are 0")
print(f"  → Residual F_react = A_react*(y-y0)_react - h*0 = 0")
print(f"    (since y0_react=0 and A_react=0)")
print(f"  → The SSN may see a trivially satisfied reaction block")
print(f"     but the PHYSICAL block still has residual from body force")

# ── TEST: Full solve with row-equilibration fix ──
print(f"\n{'='*70}")
print("FULL SOLVE TEST (with row-equilibration fix)")
print(f"{'='*70}")

import solve_nivp

solver_opts_contact = {
    'max_iter': 20,
    'adaptive_lam': True,
    'lam_update_strategy': 'none',
    'globalization': 'none',
    'use_broyden': False,
    'linear_solver': 'splu',
    'sparse': 'auto',
    'precond_reuse_steps': 50,
    'petsc_options': {'ksp_type': 'preonly', 'pc_type': 'lu'},
    'rhs_jac': cs.rhs_jac,
}

adaptive_opts = {
    'h0': tmax / 10,
    'h_max': tmax / 10,
    'atol': [1e-6] * len(cs.component_slices),
    'skip_error_indices': list(range(len(component_slices), len(cs.component_slices))),
}

nl_atol = [1e-6] * len(cs.component_slices)

try:
    t_vals, y_vals, *_ = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0, tmax),
        y0=cs.y0,
        method='backward_euler',
        projection=cs.projection,
        solver='semismooth_newton',
        projection_opts={},
        solver_opts=solver_opts_contact,
        adaptive=True,
        adaptive_opts=adaptive_opts,
        integrator_opts=cs.integrator_opts,
        rtol=1e0,
        nl_atol=nl_atol,
        nl_rtol=1e-6,
        component_slices=cs.component_slices,
        verbose=True,
        A=cs.A,
    )
    print(f"\n{'='*50}")
    print(f"Success! {len(t_vals)} time steps, t_final={t_vals[-1]:.6f} (target: {tmax:.6f})")
    print(f"  ||y_final|| = {np.linalg.norm(y_vals[-1]):.6e}")
    print(f"  y_final[:5] = {y_vals[-1][:5]}")
except Exception as e:
    print(f"\nFAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print("\nDone!")
sys.exit(0)

# Check poro.A and poro.M diagonal structure in detail  
A_dense = poro.A.toarray() if sp.issparse(poro.A) else np.array(poro.A)
M_dense = poro.M.toarray() if sp.issparse(poro.M) else np.array(poro.M)

print(f"\nporo.A diagonal analysis (N_orig={N_orig}):")
A_diag = A_dense.diagonal()
for block_name, sl in [('pressure bulk', slice(0, Np)),
                        ('u bulk', slice(Np, Np+Nu)),
                        ('lam_q', lam_q_sl),
                        ('lam_s', lam_s_sl)]:
    d = A_diag[sl]
    nz = np.count_nonzero(d)
    print(f"  {block_name:20s} [{sl.start}:{sl.stop}]: "
          f"{nz}/{sl.stop-sl.start} nonzero, "
          f"range=[{d.min():.4e}, {d.max():.4e}]")

print(f"\nporo.M diagonal analysis:")
M_diag = M_dense.diagonal()
for block_name, sl in [('pressure bulk', slice(0, Np)),
                        ('u bulk', slice(Np, Np+Nu)),
                        ('lam_q', lam_q_sl),
                        ('lam_s', lam_s_sl)]:
    d = M_diag[sl]
    nz = np.count_nonzero(d)
    print(f"  {block_name:20s} [{sl.start}:{sl.stop}]: "
          f"{nz}/{sl.stop-sl.start} nonzero, "
          f"range=[{d.min():.4e}, {d.max():.4e}]")

# Check interface DOF blocks separately
# The column transform creates: [p_b, u_b, {{p}}, [[p]], {{u}}, [[u]], multipliers...]
# Let's find exact block boundaries from component_slices
print(f"\nComponent slices:")
for i, sl in enumerate(component_slices):
    name = f"block_{i}"
    d_A = A_diag[sl]
    d_M = M_diag[sl]
    print(f"  [{i}] {sl.start:4d}:{sl.stop:4d} (size {sl.stop-sl.start:3d}) "
          f"A_diag:[{d_A.min():+.3e},{d_A.max():+.3e}] "
          f"M_diag:[{d_M.min():+.3e},{d_M.max():+.3e}]")

# Check the J_res = M/h - J_rhs = M/h + A  more carefully
h = tmax / 10
J_res_d = M_dense / h + A_dense
print(f"\nJ_res = M/h + A (h={h:.6e}):")
J_res_diag = J_res_d.diagonal()
for block_name, sl in [('pressure bulk', slice(0, Np)),
                        ('u bulk', slice(Np, Np+Nu)),
                        ('lam_q', lam_q_sl),
                        ('lam_s', lam_s_sl)]:
    d = J_res_diag[sl]
    nz = np.sum(np.abs(d) > 1e-20)
    print(f"  {block_name:20s}: {nz}/{sl.stop-sl.start} nonzero, "
          f"range=[{d.min():.4e}, {d.max():.4e}]")

# Check rank of J_res
print(f"\nJ_res rank analysis:")
svals = np.linalg.svd(J_res_d, compute_uv=False)
print(f"  rank (tol=1e-10): {np.sum(svals > 1e-10)}/{N_orig}")
print(f"  smallest 10 svals: {svals[-10:]}")
print(f"  condition number: {svals[0]/svals[-1] if svals[-1] > 0 else 'inf'}")

# Now check what happens with the AUGMENTED J_res 
J_rhs_full = cs.rhs_jac(h, cs.y0, cs.y0, h)
if sp.issparse(J_rhs_full):
    J_rhs_d = J_rhs_full.toarray()
else:
    J_rhs_d = np.array(J_rhs_full)

A_aug_d = cs.A.toarray() if sp.issparse(cs.A) else np.array(cs.A)
J_res_aug = A_aug_d / h - J_rhs_d

print(f"\nAugmented J_res = A_aug/h - J_rhs:")
print(f"  Shape: {J_res_aug.shape}")
svals_aug = np.linalg.svd(J_res_aug, compute_uv=False)
print(f"  rank (tol=1e-10): {np.sum(svals_aug > 1e-10)}/{N_aug}")
print(f"  smallest 10 svals: {svals_aug[-10:]}")

# Check the diagonal of J_rhs to understand why 151 are zero
print(f"\nJ_rhs (jac_aug) diagonal:")
J_rhs_diag = J_rhs_d.diagonal()
for i in range(min(N_aug, 30)):
    if J_rhs_diag[i] != 0:
        print(f"  [{i}] = {J_rhs_diag[i]:+.6e}")
nz_jrhs_diag = np.sum(np.abs(J_rhs_diag) > 1e-20)
print(f"  Total nonzero: {nz_jrhs_diag}/{N_aug}")

# What about the jac_aug detailed structure?
print(f"\nJ_rhs block norms:")
print(f"  [phys, phys] = {np.linalg.norm(J_rhs_d[:N_orig, :N_orig]):.4e}")
print(f"  [phys, react] = {np.linalg.norm(J_rhs_d[:N_orig, N_orig:]):.4e}")
print(f"  [react, phys] = {np.linalg.norm(J_rhs_d[N_orig:, :N_orig]):.4e}")
print(f"  [react, react] = {np.linalg.norm(J_rhs_d[N_orig:, N_orig:]):.4e}")

# Check -poro.A diagonal specifically  
neg_A_diag = (-A_dense).diagonal()
print(f"\n-poro.A diagonal (first 50):")
for i in range(50):
    if abs(neg_A_diag[i]) > 1e-20:
        print(f"  [{i}] = {neg_A_diag[i]:+.6e}")
nz_A_diag = np.sum(np.abs(neg_A_diag) > 1e-20)
print(f"  Total nonzero A diag: {nz_A_diag}/{N_orig}")

# Check if A has the expected structure
print(f"\n||A[p,p]|| = {np.linalg.norm(A_dense[:Np, :Np]):.4e}")
print(f"||A[p,u]|| = {np.linalg.norm(A_dense[:Np, Np:Np+Nu]):.4e}")
print(f"||A[u,p]|| = {np.linalg.norm(A_dense[Np:Np+Nu, :Np]):.4e}")
print(f"||A[u,u]|| = {np.linalg.norm(A_dense[Np:Np+Nu, Np:Np+Nu]):.4e}")

# Check u-block diagonal
A_uu_diag = A_dense[Np:Np+Nu, Np:Np+Nu].diagonal()
print(f"\nA[u,u] diagonal:")
nz_uu = np.sum(np.abs(A_uu_diag) > 1e-20)
print(f"  {nz_uu}/{Nu} nonzero")
print(f"  range: [{A_uu_diag.min():.4e}, {A_uu_diag.max():.4e}]")
print(f"  First 10: {A_uu_diag[:10]}")

# Check if A rows are sparse — how many zero rows?
print(f"\nporo.A row analysis (which rows are entirely zero?):")
A_csr = poro.A.tocsr() if sp.issparse(poro.A) else sp.csr_matrix(poro.A)
zero_rows = []
for i in range(N_orig):
    row_nnz = A_csr.indptr[i+1] - A_csr.indptr[i]
    if row_nnz == 0:
        zero_rows.append(i)
print(f"  Zero rows: {len(zero_rows)} → {zero_rows}")

# Check which DIAGONAL entries are nonzero (not just approximately)
nz_diag_indices = np.where(np.abs(A_diag) > 1e-20)[0]
print(f"\nNonzero A diagonal indices: {nz_diag_indices}")
print(f"  Values: {A_diag[nz_diag_indices]}")

# The key question: for u_bulk DOFs that SHOULD have K_uu diagonal,
# where did the diagonal go? Let's check A_orig before transform
# by looking at the raw off-diagonal structure
u_range = slice(Np, Np+Nu)
A_uu = A_dense[u_range, u_range]
print(f"\nA[u,u] block ({Nu}×{Nu}):")
print(f"  nnz (abs > 1e-20): {np.sum(np.abs(A_uu) > 1e-20)}")
print(f"  Frobenius norm: {np.linalg.norm(A_uu):.4e}")
print(f"  Rank: {np.linalg.matrix_rank(A_uu)}")

# Check M similarly
M_uu = M_dense[u_range, u_range]
print(f"\nM[u,u] block ({Nu}×{Nu}):")
print(f"  nnz (abs > 1e-20): {np.sum(np.abs(M_uu) > 1e-20)}")
print(f"  Frobenius norm: {np.linalg.norm(M_uu):.4e}")
print(f"  Rank: {np.linalg.matrix_rank(M_uu)}")

# Check the FIRST component slice (u_bulk) more carefully
u_bulk_sl = component_slices[1]  # should be displacement bulk
A_ub = A_dense[u_bulk_sl, u_bulk_sl]
print(f"\nA[u_bulk, u_bulk] block ({u_bulk_sl.stop-u_bulk_sl.start}×{u_bulk_sl.stop-u_bulk_sl.start}):")
print(f"  Diagonal: {np.count_nonzero(A_ub.diagonal())}/{u_bulk_sl.stop-u_bulk_sl.start} nonzero")
print(f"  Rank: {np.linalg.matrix_rank(A_ub)}")
print(f"  ||A_ub|| = {np.linalg.norm(A_ub):.4e}")
print(f"  nzcount = {np.sum(np.abs(A_ub) > 1e-20)}")

# Check the combined (M/h + A) conditioning for physical DOFs only
phys_sl = slice(0, 122)  # first 122 DOFs (physical, no multipliers)
J_phys = (M_dense[phys_sl, phys_sl] / h + A_dense[phys_sl, phys_sl])
print(f"\n(M/h + A)[phys, phys] ({phys_sl.stop}×{phys_sl.stop}):")
svals_phys = np.linalg.svd(J_phys, compute_uv=False)
print(f"  Rank: {np.sum(svals_phys > 1e-10)}")
print(f"  Smallest 5 svals: {svals_phys[-5:]}")
print(f"  Cond: {svals_phys[0]/max(svals_phys[-1], 1e-30):.2e}")
print(f"  Diagonal nonzeros: {np.count_nonzero(J_phys.diagonal())}/{phys_sl.stop}")

# Check if the issue is specifically about the DIAGONAL or the whole matrix  
# For the physical block, try actually solving a test system
b_test = np.random.randn(122)
try:
    x_test = np.linalg.solve(J_phys, b_test)
    print(f"  Test solve: ||x|| = {np.linalg.norm(x_test):.4e}  (success)")
except np.linalg.LinAlgError:
    print(f"  Test solve: SINGULAR")

print("\nDone!")
