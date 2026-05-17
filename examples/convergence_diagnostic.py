#!/usr/bin/env python3
"""
Convergence diagnostic for the prestressed fault dynamic simulation.
Instruments _solve_newton_identity to capture per-iteration residual history.
"""
import sys, time, numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── 1. Import the demo builder ──
sys.path.insert(0, "/home/david/Documents/Solve_ivp_ns/examples")
import sliding_block_one_step_patch_test_alart_curnier as _ac_demo
import solve_nivp

# ── 2. η_fluid monkey-patch (same as notebook cell 4: source rewrite) ──
import importlib, textwrap, inspect

importlib.reload(_ac_demo)

_orig_src = inspect.getsource(_ac_demo.build_demo_contact_system)
_patched_src = textwrap.dedent(_orig_src).replace(
    'eta_fluid = 2.0e-10 / 3600.0',
    'eta_fluid = 2.0e-18 / 3600.0',
)
assert 'eta_fluid = 2.0e-18 / 3600.0' in _patched_src, 'patch target not found'
_ns = vars(_ac_demo).copy()
exec(compile(_patched_src, '<eta_fluid_patch>', 'exec'), _ns)
_ac_demo.build_demo_contact_system = _ns['build_demo_contact_system']
build_demo_contact_system = _ac_demo.build_demo_contact_system
print("eta_fluid patched: 2e-10 -> 2e-18  (T_c ~ T_wave)")

make_symmetric_biaxial_rate_bc = _ac_demo.make_symmetric_biaxial_rate_bc
make_fault_prestress_patch_callbacks = _ac_demo.make_fault_prestress_patch_callbacks
make_constant_contact_offset_callbacks = _ac_demo.make_constant_contact_offset_callbacks
run_time_history_case = _ac_demo.run_time_history_case

# ── 3. Parameters (same as notebook cell 7) ──
loading_mode = "biaxial"
exx_rate = 0.0
eyy_rate = 0.0
n_elem = 20
element_type = "tri"
crack_theta = np.pi / 4.0
crack_x0 = 0.0
crack_y0 = 0.0
crack_length = 0.6
mu_friction = 0.6
initial_gap_phys = 0.0
reverse_gap_sign = False
rho_g = 0.0
normal_prestress = 6.6
background_ratio = 0.98
preload_tangential_raw = 0.0
nucleation_mode = "overstress"
patch_ratio = 1.01
patch_half_width = 0.03
patch_mu_ratio = 0.944
offset_coupling_mode = "incremental_reference"
dynamic = True
dynamic_density = 1.101
dynamic_lumped_mass = False
bulk_mu_v = 0.0
bulk_lam_v = 0.0
time_method = "backward_euler"
t_end_hours = 2.5e-5
n_steps = 500
solver_overrides = {}
integrator_overrides = {}

base_ac_options = {
    "rho_n": 1.0,
    "rho_t": 1.0,
    "gap_tol": 1.0e-12,
}

# ── 4. Build the system ──
bc_full = make_symmetric_biaxial_rate_bc(exx_rate=exx_rate, eyy_rate=eyy_rate, dp_rate=0.0)

import contextlib, io as _io
with contextlib.redirect_stdout(_io.StringIO()):
    ctx_preview = build_demo_contact_system(
        mu_friction=mu_friction, initial_gap_phys=initial_gap_phys,
        reverse_gap_sign=reverse_gap_sign, rho_g=rho_g, n_elem=n_elem,
        element_type=element_type, crack_theta=crack_theta,
        crack_x0=crack_x0, crack_y0=crack_y0, crack_length=crack_length,
        bc_full=bc_full,
        contact_backend_opts={**base_ac_options, **make_constant_contact_offset_callbacks(
            normal_offset=normal_prestress, tangential_offset=preload_tangential_raw)},
        solver_overrides=solver_overrides, time_method=time_method,
        integrator_overrides=integrator_overrides, dynamic=dynamic,
        dynamic_density=dynamic_density, dynamic_lumped_mass=dynamic_lumped_mass,
        bulk_mu_v=bulk_mu_v, bulk_lam_v=bulk_lam_v,
    )

time_scale = ctx_preview["poro"].get_scales()[0]
contact_s = np.asarray(ctx_preview["contact_plot"]["contact_s"], dtype=float)

prestress = make_fault_prestress_patch_callbacks(
    contact_s, normal_prestress=normal_prestress, mu_friction=mu_friction,
    background_ratio=background_ratio, patch_ratio=patch_ratio,
    patch_half_width=patch_half_width,
)
alart_curnier_options = {
    **base_ac_options,
    "offset_coupling_mode": offset_coupling_mode,
    "get_s0": prestress["get_s0"],
    "get_w0": prestress["get_w0"],
}

print("Building contact system...")
ctx = build_demo_contact_system(
    mu_friction=mu_friction, initial_gap_phys=initial_gap_phys,
    reverse_gap_sign=reverse_gap_sign, rho_g=rho_g, n_elem=n_elem,
    element_type=element_type, crack_theta=crack_theta,
    crack_x0=crack_x0, crack_y0=crack_y0, crack_length=crack_length,
    bc_full=bc_full, contact_backend_opts=alart_curnier_options,
    solver_overrides=solver_overrides, time_method=time_method,
    integrator_overrides=integrator_overrides, dynamic=dynamic,
    dynamic_density=dynamic_density, dynamic_lumped_mass=dynamic_lumped_mass,
    bulk_mu_v=bulk_mu_v, bulk_lam_v=bulk_lam_v,
)

cs = ctx["cs"]
ndof = cs.y0.size
print(f"Augmented DOFs: {ndof}")

# ── 5. Instrument _solve_newton_identity for convergence tracking ──
import solve_nivp.nonlinear_solvers as _ns_mod
_SolverClass = _ns_mod.ImplicitEquationSolver

# Storage for convergence data per step
_step_data = []  # list of dicts with per-step info

_orig_newton_id = _SolverClass._solve_newton_identity

def _instrumented_newton_identity(self, func, y0):
    """Wraps _solve_newton_identity to capture per-iteration residual history."""
    y = y0.copy()
    n = len(y)
    sparse_active = self._sparse_active(n)

    # Seed from persistent cache (same logic as original)
    J_local = None
    lu_local = self._lu if (self._lu is not None and self._lu_shape == (n, n)) else None
    _use_splu_path = (self.linear_solver == 'splu')
    if not _use_splu_path and self._J_cross_call is not None:
        if self._J_cross_call.shape == (n, n):
            J_local = self._J_cross_call

    # Track convergence history
    step_info = {
        'errF_history': [],
        'need_J_history': [],
        'had_cached_lu': lu_local is not None,
        'had_cached_J': J_local is not None,
    }

    prev_errF = np.inf

    for iteration in range(1, self.max_iter + 1):
        F_in = func(y)
        self.last_Fk_val = F_in
        converged, errF = self._converged_with_metric(F_in, y)

        step_info['errF_history'].append(float(errF))

        if converged:
            step_info['converged'] = True
            step_info['iters'] = iteration
            _step_data.append(step_info)
            return (y.copy(), F_in, errF, True, iteration)

        need_J = (
            (J_local is None and lu_local is None)
            or errF > 0.5 * prev_errF
        )
        step_info['need_J_history'].append(need_J)
        prev_errF = errF

        # Call original logic for the actual solve step
        # (We can't easily replicate all the solve internals, so we
        #  delegate to the original after recording diagnostics.)
        # Actually, we need the FULL original. Let's just record and
        # call through.
        break  # Exit our tracking loop; we'll call the original below

    # We only tracked the first iteration for diagnostics on the J rebuild.
    # Now call the actual original method to get the real result.
    # BUT - calling the original from scratch means we lose state. Instead,
    # let's use a cleaner approach: monkey-patch _converged_with_metric
    # to intercept every errF.

    # RESET: call the original directly
    _step_data.pop() if step_info in _step_data else None
    result = _orig_newton_id(self, func, y0)
    return result

# Actually, a cleaner approach: instrument _converged_with_metric and
# _compute_jacobian_csr to track convergence without modifying the Newton loop.

# Let's use a different, cleaner instrumentation strategy:
# Hook _converged_with_metric to record errF during each Newton iteration,
# and hook solve() to mark step boundaries.

# Reset _step_data
_step_data.clear()

# -- Hook 1: Track every errF evaluation inside Newton --
_errF_trace = []  # (errF, converged) for every _converged_with_metric call
_orig_converged = _SolverClass._converged_with_metric

def _hooked_converged(self, F, y):
    result = _orig_converged(self, F, y)
    conv, errF = result
    _errF_trace.append((float(errF), bool(conv)))
    return result

_SolverClass._converged_with_metric = _hooked_converged

# -- Hook 2: Track every Jacobian evaluation --
_jac_count_trace = []  # timestamps of Jac evaluations
_orig_compute_jac = _SolverClass._compute_jacobian_csr

def _hooked_compute_jac(self, *args, **kwargs):
    _jac_count_trace.append(time.perf_counter())
    return _orig_compute_jac(self, *args, **kwargs)

_SolverClass._compute_jacobian_csr = _hooked_compute_jac

# -- Hook 3: Track solve() calls (= one per time step) and capture result --
_solve_results = []  # (converged, iters, errF, wall_time)
_orig_solve = _SolverClass.solve

# Track which errF entries belong to which step
_errF_step_starts = []

def _hooked_solve(self, func, y0):
    _errF_step_starts.append(len(_errF_trace))
    t0 = time.perf_counter()
    result = _orig_solve(self, func, y0)
    dt = time.perf_counter() - t0
    # result is (y, F, errF, success, k)
    if isinstance(result, tuple) and len(result) >= 5:
        _solve_results.append({
            'converged': bool(result[3]),
            'iters': int(result[4]),
            'final_errF': float(result[2]),
            'wall_time': dt,
        })
    return result

_SolverClass.solve = _hooked_solve

# ── 6. Run the simulation ──
dt_nd = (t_end_hours / time_scale) / n_steps
sopts = dict(ctx["solver_opts_contact"])
sopts["linear_solver"] = "splu"  # Use standard SPLU (baseline)

N_DIAG = 500
print(f"\n{'='*60}")
print(f"Running {N_DIAG}-step convergence diagnostic...")
print(f"{'='*60}")

t0_wall = time.perf_counter()
out = solve_nivp.solve_nivp(
    fun=cs.rhs, t_span=(0.0, N_DIAG * dt_nd), y0=cs.y0.copy(),
    method=time_method, A=cs.A, h0=dt_nd, adaptive=False,
    solver="semismooth_newton",
    projection=cs.projection, component_slices=cs.component_slices,
    integrator_opts=cs.integrator_opts, solver_opts=sopts,
    nl_atol=ctx["nl_atol_contact"], nl_rtol=1.0e-6)
dt_total = time.perf_counter() - t0_wall

# Restore original methods
_SolverClass._converged_with_metric = _orig_converged
_SolverClass._compute_jacobian_csr = _orig_compute_jac
_SolverClass.solve = _orig_solve

# ── 7. Analyze convergence ──
print(f"\n{'='*60}")
print(f"CONVERGENCE ANALYSIS ({N_DIAG} steps)")
print(f"{'='*60}")
print(f"Wall time: {dt_total:.1f}s  ({dt_total/N_DIAG*1e3:.1f}ms/step)")
print(f"Total solve() calls: {len(_solve_results)}")
print(f"Total errF evaluations: {len(_errF_trace)}")
print(f"Total Jacobian evaluations: {len(_jac_count_trace)}")

converged_arr = np.array([r['converged'] for r in _solve_results], dtype=bool)
iters_arr = np.array([r['iters'] for r in _solve_results])
errF_arr = np.array([r['final_errF'] for r in _solve_results])
wall_arr = np.array([r['wall_time'] for r in _solve_results])

n_steps_actual = len(_solve_results)
n_ok = int(converged_arr.sum())
n_fail = int((~converged_arr).sum())

print(f"\nConverged: {n_ok} / {n_steps_actual}  ({n_ok/n_steps_actual*100:.1f}%)")
print(f"Failed:    {n_fail} / {n_steps_actual}  ({n_fail/n_steps_actual*100:.1f}%)")

if n_ok > 0:
    ok_mask = converged_arr
    print(f"\n--- Converged steps ---")
    print(f"  Iterations: mean={iters_arr[ok_mask].mean():.1f}, "
          f"median={np.median(iters_arr[ok_mask]):.0f}, "
          f"max={iters_arr[ok_mask].max()}, "
          f"min={iters_arr[ok_mask].min()}")
    print(f"  Final errF: mean={errF_arr[ok_mask].mean():.2e}, "
          f"max={errF_arr[ok_mask].max():.2e}")
    print(f"  Wall time:  mean={wall_arr[ok_mask].mean()*1e3:.1f}ms, "
          f"max={wall_arr[ok_mask].max()*1e3:.1f}ms, "
          f"total={wall_arr[ok_mask].sum():.1f}s")

if n_fail > 0:
    fail_mask = ~converged_arr
    print(f"\n--- Failed steps ---")
    print(f"  Iterations: mean={iters_arr[fail_mask].mean():.1f}, "
          f"max={iters_arr[fail_mask].max()}")
    print(f"  Final errF: mean={errF_arr[fail_mask].mean():.2e}, "
          f"median={np.median(errF_arr[fail_mask]):.2e}, "
          f"max={errF_arr[fail_mask].max():.2e}, "
          f"min={errF_arr[fail_mask].min():.2e}")
    print(f"  Wall time:  mean={wall_arr[fail_mask].mean()*1e3:.1f}ms, "
          f"total={wall_arr[fail_mask].sum():.1f}s")

# ── 8. Per-step residual convergence histories ──
_errF_step_starts.append(len(_errF_trace))  # sentinel for last step

print(f"\n{'='*60}")
print(f"PER-STEP RESIDUAL HISTORIES")
print(f"{'='*60}")

# Collect per-step errF histories
step_histories = []
for i in range(len(_solve_results)):
    start = _errF_step_starts[i]
    end = _errF_step_starts[i + 1] if i + 1 < len(_errF_step_starts) else len(_errF_trace)
    errFs = [e[0] for e in _errF_trace[start:end]]
    step_histories.append(errFs)

# Classify convergence patterns
n_monotone = 0
n_oscillating = 0
n_stagnating = 0
n_diverging = 0

for i, hist in enumerate(step_histories):
    if len(hist) < 2:
        continue
    # Check if monotonically decreasing (good Newton behavior)
    diffs = np.diff(hist)
    if np.all(diffs < 0):
        n_monotone += 1
    elif hist[-1] > hist[0] * 10:
        n_diverging += 1
    elif len(hist) >= 5 and abs(hist[-1] - hist[-3]) / max(abs(hist[-3]), 1e-30) < 0.1:
        n_stagnating += 1
    else:
        n_oscillating += 1

print(f"\nConvergence patterns:")
print(f"  Monotone decreasing:  {n_monotone}")
print(f"  Oscillating:          {n_oscillating}")
print(f"  Stagnating:           {n_stagnating}")
print(f"  Diverging:            {n_diverging}")

# ── 9. Show iteration histogram ──
print(f"\n{'='*60}")
print(f"ITERATION COUNT HISTOGRAM")
print(f"{'='*60}")
max_iter_val = int(iters_arr.max())
for niter in range(1, max_iter_val + 1):
    count = int((iters_arr == niter).sum())
    if count > 0:
        bar = '#' * min(count, 80)
        conv_at_niter = int(((iters_arr == niter) & converged_arr).sum())
        fail_at_niter = int(((iters_arr == niter) & ~converged_arr).sum())
        print(f"  {niter:3d} iters: {count:4d} steps  "
              f"(ok={conv_at_niter}, fail={fail_at_niter})  {bar}")

# ── 10. Show failed step details ──
if n_fail > 0:
    print(f"\n{'='*60}")
    print(f"FAILED STEP DETAILS (first 20)")
    print(f"{'='*60}")
    fail_indices = np.where(~converged_arr)[0]
    for idx in fail_indices[:20]:
        hist = step_histories[idx]
        hist_str = " -> ".join(f"{e:.2e}" for e in hist[:8])
        if len(hist) > 8:
            hist_str += f" -> ... -> {hist[-1]:.2e}"
        print(f"  Step {idx:4d}: iters={iters_arr[idx]:3d}, "
              f"final_errF={errF_arr[idx]:.2e}")
        print(f"            errF: {hist_str}")

    # Show failure distribution over time
    print(f"\nFailed step distribution (by time segment):")
    seg_size = n_steps_actual // 10
    for seg in range(10):
        s = seg * seg_size
        e = min((seg + 1) * seg_size, n_steps_actual)
        seg_fail = int((~converged_arr[s:e]).sum())
        seg_total = e - s
        pct = seg_fail / seg_total * 100 if seg_total > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  Steps {s:4d}-{e:4d}: {seg_fail:3d}/{seg_total:3d} failed "
              f"({pct:5.1f}%)  {bar}")

# ── 11. Show convergence quality over time ──
print(f"\n{'='*60}")
print(f"CONVERGENCE QUALITY OVER TIME")
print(f"{'='*60}")

# Running average of iterations and convergence rate
window = 50
for seg_start in range(0, n_steps_actual, window):
    seg_end = min(seg_start + window, n_steps_actual)
    seg_conv = converged_arr[seg_start:seg_end]
    seg_iters = iters_arr[seg_start:seg_end]
    seg_errF = errF_arr[seg_start:seg_end]
    conv_rate = seg_conv.mean() * 100
    mean_iters = seg_iters.mean()
    mean_errF = seg_errF.mean()
    max_errF = seg_errF.max()
    print(f"  Steps {seg_start:4d}-{seg_end:4d}: "
          f"conv={conv_rate:5.1f}%, "
          f"avg_iters={mean_iters:5.1f}, "
          f"avg_errF={mean_errF:.2e}, "
          f"max_errF={max_errF:.2e}")

# ── 12. solver_opts and tolerance info ──
print(f"\n{'='*60}")
print(f"SOLVER CONFIGURATION")
print(f"{'='*60}")
print(f"  nl_atol: {ctx['nl_atol_contact']}")
print(f"  nl_rtol: 1.0e-6")
print(f"  max_iter: {sopts.get('max_iter', 'default')}")
print(f"  linear_solver: {sopts.get('linear_solver', 'default')}")
print(f"  globalization: {sopts.get('globalization', 'default')}")
print(f"  sparse: {sopts.get('sparse', 'default')}")
for k, v in sorted(sopts.items()):
    if k not in ('rhs_jac', 'rhs', 'max_iter', 'linear_solver', 'globalization', 'sparse'):
        print(f"  {k}: {v}")
