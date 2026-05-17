"""Diagnostic: compare FD Jacobian of VI residual vs assembled Newton Jacobian
for the De Saxcé reduced-space backend with state-dependent friction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import scipy.sparse as sp
import solve_nivp
from solve_nivp.desaxce_contact import build_dynamic_desaxce_contact

# ── 4-DOF spring-slider with normal+tangential and state-dependent friction ──
# State: y = [vn, vt, q, slip]
#   vn   = normal velocity
#   vt   = tangential velocity
#   q    = normal displacement (gap = q, contact when q <= 0)
#   slip = accumulated slip (drives friction weakening)
# RHS:
#   dvn/dt = -k_n*q - c*vn      (normal spring + damping)
#   dvt/dt = -c*vt + F_drive    (tangential damping + driving force)
#   dq/dt  = vn                 (kinematics)
#   dslip/dt = |vt|             (slip accumulation)
# Friction: mu(slip) = mu0 + Dmu * (1 - exp(-slip/Dc))

k_n = 100.0    # normal spring stiffness
c = 1.0        # damping
F_drive = 2.0  # tangential driving force
mu0 = 0.6
Dmu = -0.15    # slip weakening
Dc = 0.1
s0 = 10.0      # normal prestress (compressive, keeps contact firmly closed)

A = np.eye(4)

def rhs(t, y):
    vn, vt, q, slip = y
    return np.array([
        -k_n*q - c*vn,
        -c*vt + F_drive,
        vn,
        np.sqrt(vt**2 + 1e-20),
    ], dtype=float)

def rhs_jac(t, y, Fk=None):
    vn, vt, q, slip = y
    J = np.zeros((4, 4), dtype=float)
    J[0, 0] = -c       # dvn_dot/dvn
    J[0, 2] = -k_n     # dvn_dot/dq
    J[1, 1] = -c       # dvt_dot/dvt
    J[2, 0] = 1.0      # dq_dot/dvn
    denom = np.sqrt(vt**2 + 1e-20)
    J[3, 1] = vt / denom  # dslip_dot/dvt
    return J

def gap_func(y, t):
    return np.array([y[2]], dtype=float)  # gap = q

def mu_func(y):
    slip = y[3]
    return mu0 + Dmu * (1.0 - np.exp(-slip / Dc))

contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=mu_func, e=0.0)]

# B maps 2D reaction [Rn, Rt] into velocity DOFs [vn, vt, q, slip]
B = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
], dtype=float)

def get_s0(y):
    return np.array([s0])

# Initial state: firmly in contact (q = -0.05), tangentially sliding
y0 = np.array([0.0, 1.0, -0.05, 0.0], dtype=float)

cs = build_dynamic_desaxce_contact(
    A=A,
    rhs_smooth=rhs,
    rhs_jac=rhs_jac,
    y0=y0,
    contacts=contacts,
    gap_func=gap_func,
    B=B,
    get_s0=get_s0,
)

print(f"n_phys = {cs.y0.size}")
print(f"projection type: {type(cs.projection).__name__}")

# ── Manual one-step solve to extract Jacobian ──────────────────────────
proj = cs.projection
n = cs.y0.size
h = 0.05
lam = 1.0  # rho scaling

# Simulate what BackwardEuler does
y_prev = cs.y0.copy()
t_new = h

# Thread context
prev_state = y_prev.copy()
step_size = h

def implicit_eq(y_new):
    return A @ ((y_new - y_prev) / h) - rhs(t_new, y_new)

def implicit_jac(y_new):
    return A / h - rhs_jac(t_new, y_new)

# Start Newton iteration from y_prev
y = y_prev.copy()

# Evaluate F(y) = implicit_eq(y)
F_y = implicit_eq(y)
candidate = y - lam * F_y

# Evaluate projection
proj_z = proj.project(
    y, candidate, rhok=lam, t=t_new, Fk_val=F_y,
    prev_state=prev_state, step_size=step_size
)

# VI residual
r_y = y - proj_z
print(f"\n=== At y0 ===")
print(f"y        = {y}")
print(f"F(y)     = {F_y}")
print(f"candidate= {candidate}")
print(f"proj_z   = {proj_z}")
print(f"r(y)     = {r_y}")
print(f"|r(y)|   = {np.linalg.norm(r_y):.6e}")

# ── Compute tangent_cone_split ─────────────────────────────────────────
Dproj, Dstate = proj.tangent_cone_split(
    candidate, y, rhok=lam, t=t_new, Fk_val=F_y,
    prev_state=prev_state, step_size=step_size,
)

Dproj_d = Dproj.toarray() if sp.issparse(Dproj) else np.asarray(Dproj)
Dstate_d = Dstate.toarray() if sp.issparse(Dstate) else np.asarray(Dstate)
J_F = implicit_jac(y)

# Assembled Newton Jacobian: J = I - Dproj - Dstate + Dproj @ (lam * J_F)
I_n = np.eye(n)
J_assembled = I_n - Dproj_d - Dstate_d + Dproj_d @ (lam * J_F)

print(f"\n=== Tangent cone split ===")
print(f"Dproj:\n{Dproj_d}")
print(f"Dstate:\n{Dstate_d}")
print(f"J_assembled:\n{J_assembled}")

# ── FD Jacobian of VI residual r(y) = y - P(y - lam*F(y), y) ──────────
eps_fd = 1e-7
J_fd = np.zeros((n, n), dtype=float)

for j in range(n):
    eps_j = eps_fd * max(1.0, abs(y[j]))
    y_p = y.copy()
    y_p[j] += eps_j

    F_p = implicit_eq(y_p)
    cand_p = y_p - lam * F_p
    proj_p = proj.project(
        y_p, cand_p, rhok=lam, t=t_new, Fk_val=F_p,
        prev_state=prev_state, step_size=step_size,
    )
    r_p = y_p - proj_p
    J_fd[:, j] = (r_p - r_y) / eps_j

print(f"\n=== FD Jacobian of r(y) ===")
print(f"J_fd:\n{J_fd}")

print(f"\n=== Jacobian comparison ===")
diff = J_assembled - J_fd
print(f"J_assembled - J_fd:\n{diff}")
print(f"|J_assembled - J_fd|_max = {np.max(np.abs(diff)):.6e}")
print(f"|J_assembled - J_fd|_F   = {np.linalg.norm(diff):.6e}")

# Relative error
J_fd_norm = np.linalg.norm(J_fd)
if J_fd_norm > 0:
    print(f"|diff|_F / |J_fd|_F      = {np.linalg.norm(diff)/J_fd_norm:.6e}")

# ── Newton step comparison ─────────────────────────────────────────────
rhs_newton = -r_y
try:
    delta_assembled = np.linalg.solve(J_assembled, rhs_newton)
    delta_fd = np.linalg.solve(J_fd, rhs_newton)
    print(f"\n=== Newton steps ===")
    print(f"delta_assembled = {delta_assembled}")
    print(f"delta_fd        = {delta_fd}")
    print(f"|delta_diff|     = {np.linalg.norm(delta_assembled - delta_fd):.6e}")

    # Check if the Newton step actually reduces the residual
    y_new_a = y + delta_assembled
    F_new_a = implicit_eq(y_new_a)
    cand_new_a = y_new_a - lam * F_new_a
    proj_new_a = proj.project(
        y_new_a, cand_new_a, rhok=lam, t=t_new, Fk_val=F_new_a,
        prev_state=prev_state, step_size=step_size,
    )
    r_new_a = y_new_a - proj_new_a

    y_new_f = y + delta_fd
    F_new_f = implicit_eq(y_new_f)
    cand_new_f = y_new_f - lam * F_new_f
    proj_new_f = proj.project(
        y_new_f, cand_new_f, rhok=lam, t=t_new, Fk_val=F_new_f,
        prev_state=prev_state, step_size=step_size,
    )
    r_new_f = y_new_f - proj_new_f

    print(f"\n=== Residual after one Newton step ===")
    print(f"|r_0|          = {np.linalg.norm(r_y):.6e}")
    print(f"|r(assembled)| = {np.linalg.norm(r_new_a):.6e}")
    print(f"|r(fd)|         = {np.linalg.norm(r_new_f):.6e}")
except np.linalg.LinAlgError as e:
    print(f"Solve failed: {e}")

# ── Full solve test ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Full solve_nivp test with SSN + tangent_cone_split")
print(f"{'='*60}")

for glob in ['none', 'linesearch']:
    print(f"\n--- globalization={glob} ---")
    t_arr, y_arr, h_arr, fk_arr, info_arr = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.1),
        y0=cs.y0,
        method="backward_euler",
        projection=cs.projection,
        solver="semismooth_newton",
        solver_opts={
            "tol": 1e-10,
            "max_iter": 40,
            "globalization": glob,
            "linear_solver": "splu",
            "rhs_jac": cs.rhs_jac,
            "adaptive_lam": False,
        },
        adaptive=False,
        h0=0.05,
        integrator_opts=cs.integrator_opts,
        component_slices=cs.component_slices,
        A=cs.A,
        store_fk=False,
    )

    print(f"  Steps taken: {len(t_arr) - 1}")
    print(f"  t_final = {t_arr[-1]:.6f}")
    for i, (err, success, iters) in enumerate(info_arr):
        print(f"  step {i}: success={success}, iters={iters}, err={err:.3e}")
