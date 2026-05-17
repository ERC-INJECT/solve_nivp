#!/usr/bin/env python3
"""
Single-step deep convergence diagnostic.
Shows per-component residual breakdown and active set changes.
"""
import sys, time, numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/home/david/Documents/Solve_ivp_ns/examples")
import sliding_block_one_step_patch_test_alart_curnier as _ac_demo
import solve_nivp

# η_fluid monkey-patch
import importlib, textwrap, inspect
importlib.reload(_ac_demo)
_orig_src = inspect.getsource(_ac_demo.build_demo_contact_system)
_patched_src = textwrap.dedent(_orig_src).replace(
    'eta_fluid = 2.0e-10 / 3600.0', 'eta_fluid = 2.0e-18 / 3600.0')
assert 'eta_fluid = 2.0e-18 / 3600.0' in _patched_src
_ns = vars(_ac_demo).copy()
exec(compile(_patched_src, '<eta_fluid_patch>', 'exec'), _ns)
_ac_demo.build_demo_contact_system = _ns['build_demo_contact_system']

import contextlib, io as _io

# ── Parameters (same as notebook cell 7) ──
bc_full = _ac_demo.make_symmetric_biaxial_rate_bc(exx_rate=0.0, eyy_rate=0.0, dp_rate=0.0)

common_kw = dict(
    mu_friction=0.6, initial_gap_phys=0.0, reverse_gap_sign=False,
    rho_g=0.0, n_elem=20, element_type="tri",
    crack_theta=np.pi/4, crack_x0=0.0, crack_y0=0.0, crack_length=0.6,
    bc_full=bc_full, time_method="backward_euler",
    solver_overrides={}, integrator_overrides={},
    dynamic=True, dynamic_density=1.101, dynamic_lumped_mass=False,
    bulk_mu_v=0.0, bulk_lam_v=0.0,
)
base_ac = {"rho_n": 1.0, "rho_t": 1.0, "gap_tol": 1e-12}

with contextlib.redirect_stdout(_io.StringIO()):
    ctx_preview = _ac_demo.build_demo_contact_system(
        contact_backend_opts={**base_ac, **_ac_demo.make_constant_contact_offset_callbacks(
            normal_offset=6.6, tangential_offset=0.0)}, **common_kw)

time_scale = ctx_preview["poro"].get_scales()[0]
contact_s = np.asarray(ctx_preview["contact_plot"]["contact_s"], dtype=float)

prestress = _ac_demo.make_fault_prestress_patch_callbacks(
    contact_s, normal_prestress=6.6, mu_friction=0.6,
    background_ratio=0.98, patch_ratio=1.01, patch_half_width=0.03)

ac_opts = {**base_ac, "offset_coupling_mode": "incremental_reference",
           "get_s0": prestress["get_s0"], "get_w0": prestress["get_w0"]}

print("Building system...")
ctx = _ac_demo.build_demo_contact_system(contact_backend_opts=ac_opts, **common_kw)
cs = ctx["cs"]
ndof = cs.y0.size

# ── Understand component structure ──
comp_slices = cs.component_slices
print(f"\nSystem: {ndof} DOFs, {len(comp_slices)} component blocks")
nl_atol = ctx["nl_atol_contact"]
print(f"nl_atol has {len(nl_atol)} entries: {nl_atol}")

block_names = []
for i, sl in enumerate(comp_slices):
    if isinstance(sl, slice):
        sz = sl.stop - sl.start
    else:
        sz = len(np.asarray(sl))
    name = f"block_{i}"
    if i < len(nl_atol):
        tol = nl_atol[i]
    else:
        tol = "N/A"
    block_names.append(name)
    print(f"  Block {i}: {sz:5d} DOFs, tol={tol}")

# ── Get the RHS and Jacobian functions ──
rhs_jac = ctx["solver_opts_contact"]["rhs_jac"]

# ── Manual single-step Newton to track per-component residuals ──
dt_nd = (2.5e-5 / time_scale) / 500
y0 = cs.y0.copy()

# The implicit equation for backward Euler: F(y) = A*(y - y0)/h - rhs(t_new, y)
# But solve_nivp wraps this. Let's just instrument a few steps of the actual solver.
import solve_nivp.nonlinear_solvers as _ns_mod
_SolverClass = _ns_mod.ImplicitEquationSolver

# Storage
_iter_residuals = []  # list of (iteration, per_block_errF)
_iter_full_F = []     # store the actual F vector at each iteration

_orig_converged = _SolverClass._converged_with_metric

def _deep_converged(self, F, y):
    result = _orig_converged(self, F, y)
    conv, errF = result
    
    # Compute per-block residual norms
    block_norms = []
    for i, sl in enumerate(comp_slices):
        if isinstance(sl, slice):
            idx = np.arange(sl.start, sl.stop)
        else:
            idx = np.asarray(sl)
        if len(idx) > 0 and max(idx) < len(F):
            block_F = F[idx]
            block_norm = float(np.linalg.norm(block_F))
            block_max = float(np.max(np.abs(block_F)))
            n_large = int(np.sum(np.abs(block_F) > 1.0))
            block_norms.append((block_norm, block_max, n_large, len(idx)))
        else:
            block_norms.append((0.0, 0.0, 0, 0))
    
    _iter_residuals.append({
        'errF': float(errF),
        'converged': bool(conv),
        'block_norms': block_norms,
    })
    
    # Store full F for the first few iterations
    if len(_iter_full_F) < 100:
        _iter_full_F.append(F.copy())
    
    return result

_SolverClass._converged_with_metric = _deep_converged

# Run just 5 steps to get detailed per-iteration data
N_STEPS = 5
print(f"\nRunning {N_STEPS} steps with full per-iteration tracking...")
sopts = dict(ctx["solver_opts_contact"])
sopts["linear_solver"] = "splu"

out = solve_nivp.solve_nivp(
    fun=cs.rhs, t_span=(0.0, N_STEPS * dt_nd), y0=y0.copy(),
    method="backward_euler", A=cs.A, h0=dt_nd, adaptive=False,
    solver="semismooth_newton",
    projection=cs.projection, component_slices=cs.component_slices,
    integrator_opts=cs.integrator_opts, solver_opts=sopts,
    nl_atol=ctx["nl_atol_contact"], nl_rtol=1.0e-6)

_SolverClass._converged_with_metric = _orig_converged

# ── Analyze ──
print(f"\n{'='*70}")
print(f"PER-ITERATION RESIDUAL BREAKDOWN (first {N_STEPS} steps)")
print(f"{'='*70}")

# Group iterations by step (max_iter=30 per step, but may converge earlier)
iter_idx = 0
step_num = 0
while iter_idx < len(_iter_residuals):
    step_iters = []
    while iter_idx < len(_iter_residuals):
        entry = _iter_residuals[iter_idx]
        step_iters.append(entry)
        iter_idx += 1
        if entry['converged'] or len(step_iters) >= 31:  # max_iter=30 + final eval
            break
    
    conv_str = "CONVERGED" if step_iters[-1]['converged'] else "FAILED"
    print(f"\n--- Step {step_num} ({conv_str}, {len(step_iters)} evals) ---")
    
    # Show per-iteration breakdown
    header = f"{'Iter':>4s}  {'errF':>10s}"
    for i in range(len(comp_slices)):
        header += f"  {'blk'+str(i)+'-L2':>12s}  {'blk'+str(i)+'-max':>12s}"
    print(header)
    
    for k, entry in enumerate(step_iters[:10]):  # first 10 iterations
        line = f"{k+1:4d}  {entry['errF']:10.2e}"
        for bn in entry['block_norms']:
            line += f"  {bn[0]:12.2e}  {bn[1]:12.2e}"
        print(line)
    
    if len(step_iters) > 10:
        print(f"  ... ({len(step_iters) - 10} more iterations)")
        # Show last 3
        for k_off, entry in enumerate(step_iters[-3:]):
            k = len(step_iters) - 3 + k_off
            line = f"{k+1:4d}  {entry['errF']:10.2e}"
            for bn in entry['block_norms']:
                line += f"  {bn[0]:12.2e}  {bn[1]:12.2e}"
            print(line)
    
    step_num += 1
    if step_num >= N_STEPS:
        break

# ── Focus on the reaction DOFs ──
print(f"\n{'='*70}")
print(f"REACTION DOF ANALYSIS (last block)")
print(f"{'='*70}")

react_slice = comp_slices[-1]
if isinstance(react_slice, slice):
    react_idx = np.arange(react_slice.start, react_slice.stop)
else:
    react_idx = np.asarray(react_slice)

print(f"Reaction DOFs: {len(react_idx)} (indices {react_idx[0]}..{react_idx[-1]})")

# For the stored full F vectors, show which reaction DOFs are large
if len(_iter_full_F) > 0:
    print(f"\nIteration-by-iteration reaction residual (first step, first 10 iters):")
    for k in range(min(10, len(_iter_full_F))):
        F = _iter_full_F[k]
        react_F = F[react_idx]
        print(f"  Iter {k+1}: react L2={np.linalg.norm(react_F):.2e}, "
              f"max={np.max(np.abs(react_F)):.2e}, "
              f"n>1={np.sum(np.abs(react_F) > 1.0)}, "
              f"n>100={np.sum(np.abs(react_F) > 100.0)}")
        
        # Show top 5 worst reaction DOFs
        worst = np.argsort(np.abs(react_F))[-5:][::-1]
        for w in worst:
            global_idx = react_idx[w]
            print(f"    react[{w}] (dof {global_idx}): F={react_F[w]:.4e}")

# ── Compare reaction vs non-reaction residuals ──
print(f"\n{'='*70}")
print(f"NON-REACTION vs REACTION RESIDUAL COMPARISON")
print(f"{'='*70}")

non_react_idx = np.arange(0, react_idx[0])
for k in range(min(10, len(_iter_full_F))):
    F = _iter_full_F[k]
    nr_norm = np.linalg.norm(F[non_react_idx])
    r_norm = np.linalg.norm(F[react_idx])
    print(f"  Iter {k+1}: non-react L2={nr_norm:.2e}, react L2={r_norm:.2e}, "
          f"ratio(react/non-react)={r_norm/max(nr_norm, 1e-30):.1f}")
