#!/usr/bin/env python3
"""
Standalone timing diagnostic for the prestressed fault dynamic simulation.
Instruments splu, Jacobian, and RHS calls to find the performance bottleneck.
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

# Preview system for contact_s
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
print(f"Augmented DOFs: {cs.y0.size}")
print(f"A sparse? {sp.issparse(cs.A)}, format={getattr(cs.A, 'format', 'dense')}, nnz={getattr(cs.A, 'nnz', 'N/A')}")

# ── 5. Instrument splu, Jacobian, RHS, and per-step timing ──
_real_splu = spla.splu
_splu_calls = []

def _splu_hook(*a, **kw):
    t0 = time.perf_counter()
    r = _real_splu(*a, **kw)
    _splu_calls.append(time.perf_counter() - t0)
    return r

spla.splu = _splu_hook

# Also instrument np.linalg.solve to detect any dense fallback
_real_np_solve = np.linalg.solve
_dense_solve_calls = []

def _dense_solve_hook(a, b):
    t0 = time.perf_counter()
    r = _real_np_solve(a, b)
    _dense_solve_calls.append(time.perf_counter() - t0)
    return r

np.linalg.solve = _dense_solve_hook

_orig_jac = ctx["solver_opts_contact"]["rhs_jac"]
_orig_rhs = cs.rhs
_jac_calls = []
_rhs_calls = []

def _jac_hook(*a, **kw):
    t0 = time.perf_counter()
    r = _orig_jac(*a, **kw)
    _jac_calls.append(time.perf_counter() - t0)
    return r

def _rhs_hook(*a, **kw):
    t0 = time.perf_counter()
    r = _orig_rhs(*a, **kw)
    _rhs_calls.append(time.perf_counter() - t0)
    return r

# Instrument per-step timing for BOTH solver paths
import solve_nivp.nonlinear_solvers as _ns_mod
_OrigSolverClass = _ns_mod.ImplicitEquationSolver
_per_step_times = []
_per_step_iters = []
_per_step_converged = []
_per_step_splu_before = []

# Instrument semismooth Newton (the path run_time_history_case uses)
_orig_solve_ssn = _OrigSolverClass._solve_with_semismooth_newton

def _instrumented_solve_ssn(self, *a, **kw):
    splu_before = len(_splu_calls)
    t0 = time.perf_counter()
    result = _orig_solve_ssn(self, *a, **kw)
    dt = time.perf_counter() - t0
    _per_step_times.append(dt)
    # result is (y, Fk, err, success, k) — 5-tuple
    if isinstance(result, tuple) and len(result) >= 5:
        _per_step_converged.append(bool(result[3]))
        _per_step_iters.append(int(result[4]))
    _per_step_splu_before.append(len(_splu_calls) - splu_before)
    return result

_OrigSolverClass._solve_with_semismooth_newton = _instrumented_solve_ssn

# Also instrument VI path as fallback
_orig_solve_vi = _OrigSolverClass._solve_vi_identity

def _instrumented_solve_vi(self, *a, **kw):
    splu_before = len(_splu_calls)
    t0 = time.perf_counter()
    result = _orig_solve_vi(self, *a, **kw)
    dt = time.perf_counter() - t0
    _per_step_times.append(dt)
    if isinstance(result, tuple) and len(result) >= 5:
        _per_step_converged.append(bool(result[3]))
        _per_step_iters.append(int(result[4]))
    _per_step_splu_before.append(len(_splu_calls) - splu_before)
    return result

_OrigSolverClass._solve_vi_identity = _instrumented_solve_vi

dt_nd = (t_end_hours / time_scale) / n_steps
sopts = dict(ctx["solver_opts_contact"])
sopts["rhs_jac"] = _jac_hook
sopts["linear_solver"] = "umfpack"  # Use UMFPACK with symbolic reuse

# ── 6. Run full 500 steps to capture failing steps ──
N_DIAG = 500
print(f"\n{'='*60}")
print(f"Running {N_DIAG}-step diagnostic...")
print(f"{'='*60}")

t0 = time.perf_counter()
out = solve_nivp.solve_nivp(
    fun=_rhs_hook, t_span=(0.0, N_DIAG * dt_nd), y0=cs.y0.copy(),
    method=time_method, A=cs.A, h0=dt_nd, adaptive=False,
    solver="semismooth_newton",
    projection=cs.projection, component_slices=cs.component_slices,
    integrator_opts=cs.integrator_opts, solver_opts=sopts,
    nl_atol=ctx["nl_atol_contact"], nl_rtol=1.0e-6)
dt_total = time.perf_counter() - t0

spla.splu = _real_splu
np.linalg.solve = _real_np_solve

# ── 7. Report ──
print(f"\n{'='*60}")
print(f"RESULTS: {N_DIAG}-step run")
print(f"{'='*60}")
print(f"Wall time:     {dt_total:.3f} s  ({dt_total/N_DIAG*1e3:.1f} ms/step)")
print(f"RHS calls:     {len(_rhs_calls):6d}  total={sum(_rhs_calls)*1e3:.1f}ms  avg={np.mean(_rhs_calls)*1e3:.2f}ms")
print(f"Jac calls:     {len(_jac_calls):6d}  total={sum(_jac_calls)*1e3:.1f}ms  avg={np.mean(_jac_calls)*1e3:.2f}ms" if _jac_calls else f"Jac calls:     0")
print(f"splu calls:    {len(_splu_calls):6d}  total={sum(_splu_calls)*1e3:.1f}ms" if _splu_calls else "splu calls:    0")
if _splu_calls:
    print(f"  splu avg: {np.mean(_splu_calls)*1e3:.2f}ms, max: {max(_splu_calls)*1e3:.2f}ms")
print(f"DENSE solves:  {len(_dense_solve_calls):6d}  total={sum(_dense_solve_calls)*1e3:.1f}ms" if _dense_solve_calls else "DENSE solves:  0  <<< GOOD: no dense fallback")

overhead = dt_total - sum(_rhs_calls) - sum(_jac_calls) - sum(_splu_calls) - sum(_dense_solve_calls)

print(f"\nBreakdown:")
print(f"  RHS eval:     {sum(_rhs_calls)/dt_total*100:5.1f}%  ({sum(_rhs_calls)*1e3:.0f}ms)")
if _jac_calls:
    print(f"  Jac build:    {sum(_jac_calls)/dt_total*100:5.1f}%  ({sum(_jac_calls)*1e3:.0f}ms)")
if _splu_calls:
    print(f"  splu fact:    {sum(_splu_calls)/dt_total*100:5.1f}%  ({sum(_splu_calls)*1e3:.0f}ms)")
if _dense_solve_calls:
    print(f"  DENSE solve:  {sum(_dense_solve_calls)/dt_total*100:5.1f}%  ({sum(_dense_solve_calls)*1e3:.0f}ms)")
print(f"  Other:        {overhead/dt_total*100:5.1f}%  ({overhead*1e3:.0f}ms)")
print(f"    (includes: LU.solve, projection, linesearch, overhead)")

# Per-call statistics
print(f"\nPer-call statistics:")
print(f"  RHS: {len(_rhs_calls)} calls × {np.mean(_rhs_calls)*1e3:.2f}ms = {sum(_rhs_calls)*1e3:.0f}ms")
print(f"  Avg RHS/step: {len(_rhs_calls)/N_DIAG:.1f}")
if _jac_calls:
    print(f"  Jac: {len(_jac_calls)} calls × {np.mean(_jac_calls)*1e3:.2f}ms = {sum(_jac_calls)*1e3:.0f}ms")
    print(f"  Avg Jac/step: {len(_jac_calls)/N_DIAG:.1f}")
if _splu_calls:
    print(f"  splu: {len(_splu_calls)} calls × {np.mean(_splu_calls)*1e3:.2f}ms = {sum(_splu_calls)*1e3:.0f}ms")

# Check output
if isinstance(out, tuple):
    t_arr = out[0]
    y_arr = out[1]
else:
    t_arr = out.t
    y_arr = out.y

print(f"\nOutput: {len(t_arr)} time points, final t = {t_arr[-1]:.6e}")

# ── 8. Per-step analysis ──
print(f"\n{'='*60}")
print(f"Per-step analysis:")
print(f"{'='*60}")
print(f"Total steps: {len(_per_step_times)}")
converged = np.array(_per_step_converged)
times = np.array(_per_step_times)
iters = np.array(_per_step_iters)
splu_per = np.array(_per_step_splu_before)

n_ok = int(converged.sum())
n_fail = int((~converged).sum())
print(f"Converged: {n_ok}, Failed: {n_fail}")

if n_ok > 0:
    ok_times = times[converged]
    ok_iters = iters[converged]
    ok_splu = splu_per[converged]
    print(f"\nConverged steps:")
    print(f"  wall: mean={np.mean(ok_times)*1e3:.2f}ms, max={np.max(ok_times)*1e3:.2f}ms")
    print(f"  iters: mean={np.mean(ok_iters):.1f}, max={np.max(ok_iters)}")
    print(f"  splu/step: mean={np.mean(ok_splu):.2f}, max={np.max(ok_splu)}")
    print(f"  total wall: {np.sum(ok_times)*1e3:.0f}ms")

if n_fail > 0:
    fail_times = times[~converged]
    fail_iters = iters[~converged]
    fail_splu = splu_per[~converged]
    print(f"\nFailed steps:")
    print(f"  wall: mean={np.mean(fail_times)*1e3:.2f}ms, max={np.max(fail_times)*1e3:.2f}ms")
    print(f"  iters: mean={np.mean(fail_iters):.1f}, max={np.max(fail_iters)}")
    print(f"  splu/step: mean={np.mean(fail_splu):.2f}, max={np.max(fail_splu)}")
    print(f"  total wall: {np.sum(fail_times)*1e3:.0f}ms")
    print(f"  % of total time: {np.sum(fail_times)/dt_total*100:.1f}%")

# Show the 10 slowest steps
print(f"\n10 slowest steps:")
slow_idx = np.argsort(times)[-10:][::-1]
for i in slow_idx:
    print(f"  step {i:3d}: {times[i]*1e3:8.1f}ms, iters={iters[i]:3d}, converged={converged[i]}, splu={splu_per[i]}")

# Restore
_OrigSolverClass._solve_vi_identity = _orig_solve_vi
_OrigSolverClass._solve_with_semismooth_newton = _orig_solve_ssn
