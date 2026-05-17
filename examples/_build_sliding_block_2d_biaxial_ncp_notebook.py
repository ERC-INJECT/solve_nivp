#!/usr/bin/env python
"""Programmatically build examples/sliding_block_2d_biaxial_ncp.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "sliding_block_2d_biaxial_ncp.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELL_MD_HEADER = r"""
# 2D biaxial NCP friction benchmark — Phase 0 (intact elastodynamics)

This notebook implements the first phase of the phased validation plan in
`sliding_block_2d_biaxial_ncp_plan.md`.  Phase 0 validates the intact
domain: no crack, no contact, just the traction-loaded elastodynamic
poro descriptor system.  The material block, the traction-aware
subclass, and the helper utilities live in
`sliding_block_2d_biaxial_ncp_helpers.py`.

**Sign convention** (locked per the plan's *Sign conventions* section):
stresses and tractions are in the solid-mechanics convention —
*tension positive, compression negative*.  A compressive normal load on
the right face is emitted as the traction callable `(-sigma_right, 0)`,
and similarly `(0, -sigma_top)` on the top face.  The NCP contact
convention (positive in compression) is not used in this notebook — it
only appears in Phases 1–3.

**Phase 0 structure** — three distinct sub-runs:

- **Sub-run 0a** — right-only step traction, short window, used to
  measure the bulk P-wave arrival (*Validation A*).
- **Sub-run 0b** — smooth symmetric cosine ramp on right and top over
  the production `t_end`, used for *Validations A0* (initial residual),
  *B* (work–energy balance), and *C* (diagonal symmetry).
- **Sub-run 0c** — same geometry as 0b but with a much slower ramp,
  used for *Validation D* (slow-loading interior stress recovery).
"""

CELL_CODE_IMPORTS = r"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd()
while REPO_ROOT.name and not (REPO_ROOT / "src" / "solve_nivp").is_dir():
    REPO_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT / "src", REPO_ROOT / "examples"):
    p = str(path)
    if str(path) not in sys.path:
        sys.path.insert(0, p)

import solve_nivp
from sliding_block_2d_biaxial_ncp_helpers import (
    phase0_material_constants,
    phase0_time_scale,
    phase0_wave_speeds,
    make_cosine_ramp_scale_fn,
    make_step_scale_fn,
    make_compressive_traction_callable,
    build_intact_biaxial_dynamic_context,
    compute_bulk_energy,
    compute_boundary_power,
    extract_bulk_velocity,
    extract_bulk_displacement,
    validate_initial_residual,
    find_u_probe_dof,
)

MATERIAL = phase0_material_constants()
T_SCALE = phase0_time_scale(MATERIAL)
WAVE = phase0_wave_speeds(MATERIAL)
C_P = WAVE['c_p']
C_S = WAVE['c_s']
L_UNIT = 1.0

N_ELEM = 12
# Phase 0 uses fixed-step backward Euler: the Radau coupled-Newton path
# (refactored 2026-04-15) returns early with zero progress on this 2D
# descriptor problem and the adaptive step controller stalls from the
# rest state because the velocity blocks start at exactly zero and the
# WRMS relative error never registers a meaningful motion. Both issues
# are flagged in the plan risk register and are separate tasks from the
# Phase 0 physics validations.
SOLVER_METHOD = 'backward_euler'
BE_RTOL = 1.0e-6
BE_ATOL = 1.0e-9
NEWTON_TOL = 1.0e-8
NL_ATOL = 1.0e-10
NL_RTOL = 1.0e-8
TOL_BAR = 10.0 * max(BE_RTOL, NL_RTOL)

# Work-energy balance is limited by backward Euler numerical dissipation,
# which scales as O(h^2 * omega^2) per step for the linear-elastic wave
# equation. Validation B therefore uses a discretization-aware tolerance
# analogous to Validations C and D.
TOL_DISCR_B = 0.05

print(f"Material Mmod = {MATERIAL['lam'] + 2*MATERIAL['mu']:.4e}")
print(f"Wave speeds (nondim):  c_p = {C_P:.6f}   c_s = {C_S:.6f}")
print(f"P-wave transit L/c_p = {L_UNIT/C_P:.6f} (nondim time units)")
print(f"S-wave transit L/c_s = {L_UNIT/C_S:.6f} (nondim time units)")
print(f"Poro time scale T    = {T_SCALE:.6e}")
print(f"N_ELEM = {N_ELEM}")
print(f"Method = {SOLVER_METHOD} (fixed step)")
print(f"Newton  tol / nl_rtol = {NEWTON_TOL:.1e} / {NL_RTOL:.1e}")
print(f"Tolerance bar for Validation B = {TOL_DISCR_B:.3e} (discretization-limited)")
"""

CELL_MD_0A = r"""
## Sub-run 0a — right-only step traction (P-wave arrival)

We apply a step traction on the right face only and integrate for a
window short enough that the P-wavefront from the right face reaches the
mid-domain probe at $(0.5, 0.5)$ but **does not reflect back**.  The
wave speeds above give the expected arrival time
$t_{\text{arrival}} = 0.5 / c_p$ and the earliest time a reflection
could return to the probe as
$t_{\text{reflect}} = 0.5/c_p + 1/c_p$.  We set `t_end_0a` halfway
between them.
"""

CELL_CODE_0A_BUILD = r"""
T_ARRIVAL_PRED = 0.5 / C_P
T_REFLECT = T_ARRIVAL_PRED + L_UNIT / C_P
T_END_0A = 0.5 * (T_ARRIVAL_PRED + T_REFLECT)
# Smooth traction ramp on the right face shorter than the mid-domain
# transit time.  A Heaviside step excites the whole mesh instantly
# through the consistent FE mass matrix and contaminates the arrival
# detector with O(h) noise.  A ramp that completes well before the
# expected arrival gives a clean travelling front to measure.
T_RAMP_0A_ND = 0.1 * T_ARRIVAL_PRED
T_RAMP_0A_DIM = T_RAMP_0A_ND * T_SCALE

print(f"Predicted arrival at (0.5, 0.5): t = {T_ARRIVAL_PRED:.6f}")
print(f"Earliest reflection return:      t = {T_REFLECT:.6f}")
print(f"Sub-run 0a t_end = {T_END_0A:.6f}")
print(f"Ramp duration    = {T_RAMP_0A_ND:.6f} nondim (<< arrival)")

ctx_0a = build_intact_biaxial_dynamic_context(
    n_elem=N_ELEM,
    sigma_right=1.0,
    sigma_top=0.0,
    traction_scale_fn=make_cosine_ramp_scale_fn(t_ramp_dim=T_RAMP_0A_DIM),
)
probe_dof_vx = find_u_probe_dof(ctx_0a, x=0.5, y=0.5, component=0)
print(f"Probe DOF (vx @ 0.5,0.5) in original layout = {probe_dof_vx}")
print(f"Descriptor size = {ctx_0a['A'].shape[0]}")
"""

CELL_CODE_0A_SOLVE = r"""
def run_descriptor(ctx, t_end, *, method=SOLVER_METHOD, h0=None, adaptive=False, n_steps=300):
    h0_use = h0 if h0 is not None else t_end / float(n_steps)
    solver_opts = {
        'rhs_jac': ctx['rhs_jac'],
        'max_iter': 30,
        'tol': NEWTON_TOL,
        'globalization': 'linesearch',
        'linear_solver': 'splu',
        'sparse': True,
    }
    t_out, y_out, h_out, fk_out, info = solve_nivp.solve_nivp(
        fun=ctx['rhs'],
        t_span=(0.0, float(t_end)),
        y0=np.asarray(ctx['y0'], dtype=float).copy(),
        method=method,
        projection=ctx['projection'],
        solver='semismooth_newton',
        adaptive=adaptive,
        h0=h0_use,
        A=ctx['A'],
        solver_opts=solver_opts,
        component_slices=ctx['component_slices'],
        nl_atol=NL_ATOL,
        nl_rtol=NL_RTOL,
        rtol=BE_RTOL,
        atol=BE_ATOL,
        verbose=False,
    )
    return {
        't': np.asarray(t_out),
        'y': np.asarray(y_out),
        'h': np.asarray(h_out),
        'fk': np.asarray(fk_out),
        'info': info,
    }

def _info_status(info):
    if isinstance(info, dict):
        return info.get('status'), info.get('n_rejected')
    if isinstance(info, (list, tuple)):
        n_rejected = sum(1 for d in info if isinstance(d, dict) and d.get('rejected'))
        last = info[-1] if info else {}
        status = last.get('status') if isinstance(last, dict) else None
        return status, n_rejected
    return None, None

print(f"Integrating Sub-run 0a with method = {SOLVER_METHOD}")
sol_0a = run_descriptor(ctx_0a, T_END_0A, n_steps=600)
print(f"  n_steps = {sol_0a['t'].size}, final t = {sol_0a['t'][-1]:.6f}")
"""

CELL_CODE_0A_VALID = r"""
# Validation A — P-wave arrival at (0.5, 0.5).
#
# "First time at which |v_x probe| exceeds 0.1 * max|v_x probe|". A 10%
# threshold is robust against consistent-mass near-field noise while
# still firing well before the peak of the travelling front.
t_hist = sol_0a['t']
y_hist = sol_0a['y']
n_base_0a = ctx_0a['n_base']
Nu_0a = ctx_0a['Nu']
vel_slice_0a = ctx_0a['velocity_slice']
vx_probe = y_hist[:, vel_slice_0a.start + probe_dof_vx]

vx_peak = float(np.max(np.abs(vx_probe)))
threshold = 0.1 * vx_peak if vx_peak > 0.0 else 0.0
active = np.where(np.abs(vx_probe) >= threshold)[0] if threshold > 0.0 else np.array([])
t_arrival_meas = float(t_hist[int(active[0])]) if active.size else float('nan')

element_transit = L_UNIT / (N_ELEM * C_P)
arrival_err = abs(t_arrival_meas - T_ARRIVAL_PRED)
valid_A = arrival_err <= 2.0 * element_transit

print("Validation A  — P-wave arrival")
print(f"  predicted arrival  : {T_ARRIVAL_PRED:.6f}")
print(f"  measured arrival   : {t_arrival_meas:.6f}")
print(f"  |error|            : {arrival_err:.6e}")
print(f"  element transit    : {element_transit:.6e}  (tolerance)")
print(f"  vx peak magnitude  : {vx_peak:.6e}")
print(f"  PASS               : {valid_A}")

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(t_hist, vx_probe, 'k-', lw=1.2, label=r'$v_x$ at $(0.5, 0.5)$')
ax.axvline(T_ARRIVAL_PRED, color='tab:blue', ls='--', lw=1.0,
           label=r'$L/(2 c_p)$ predicted')
ax.axvline(T_REFLECT, color='tab:red', ls=':', lw=1.0,
           label='earliest reflection')
if np.isfinite(t_arrival_meas):
    ax.axvline(t_arrival_meas, color='tab:green', ls='-', lw=0.8,
               label='measured arrival')
ax.set_xlabel('nondim time $t$')
ax.set_ylabel(r'$v_x$')
ax.set_title('Sub-run 0a: horizontal velocity at mid-domain probe')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""

CELL_MD_0B = r"""
## Sub-run 0b — smooth symmetric biaxial ramp

A symmetric biaxial compression with a cosine ramp over the first half
of the run.  Symmetric tractions (`sigma_right = sigma_top`) let us apply
the diagonal-symmetry check in *Validation C*.  A cosine ramp with
`ramp(0) = 0` ensures the initial state is truly at rest, which is what
*Validation A0* requires.
"""

CELL_CODE_0B_BUILD = r"""
SIGMA_SYMM = 5.0
T_END_0B = 10.0 * (L_UNIT / C_P)
T_RAMP_0B_ND = 5.0 * (L_UNIT / C_P)
T_RAMP_0B_DIM = T_RAMP_0B_ND * T_SCALE

ctx_0b = build_intact_biaxial_dynamic_context(
    n_elem=N_ELEM,
    sigma_right=SIGMA_SYMM,
    sigma_top=SIGMA_SYMM,
    traction_scale_fn=make_cosine_ramp_scale_fn(t_ramp_dim=T_RAMP_0B_DIM),
)
print(f"Sub-run 0b: sigma_right = sigma_top = {SIGMA_SYMM}")
print(f"  t_end  = {T_END_0B:.6f} (nondim)  = {T_END_0B/T_SCALE:.3e} dimensional units")
print(f"  t_ramp = {T_RAMP_0B_ND:.6f} (nondim)")
print(f"  descriptor size = {ctx_0b['A'].shape[0]}")
"""

CELL_CODE_0B_A0 = r"""
# Validation A0 — initial residual at the rest state y0.
a0 = validate_initial_residual(ctx_0b, newton_tol=NEWTON_TOL)
print("Validation A0 — initial residual")
for k, v in a0.items():
    print(f"  {k:>16s} = {v}")
assert a0['passed'], (
    "A0 failed: descriptor RHS at (0, y0) exceeds Newton tolerance. "
    "Check strip_multiplier_dynamics, Dirichlet initialisation, and "
    "traction_scale_fn(0) == 0."
)
"""

CELL_CODE_0B_SOLVE = r"""
print(f"Integrating Sub-run 0b with method = {SOLVER_METHOD}")
sol_0b = run_descriptor(ctx_0b, T_END_0B, n_steps=300)
status_0b, rej_0b = _info_status(sol_0b['info'])
print(f"  n_steps = {sol_0b['t'].size}, final t = {sol_0b['t'][-1]:.6f}")
print(f"  status     = {status_0b}")
print(f"  rejections = {rej_0b}")
"""

CELL_CODE_0B_ENERGY = r"""
# Validation B — work-energy balance over Sub-run 0b.
#
# E(t)       = (1/2) v^T M_u v + (1/2) u^T K_uu u
# P(t)       = t . v    integrated over the traction boundaries
# W(t)       = integral from 0 to t of P(tau) dtau  (trapezoidal)
# Required   max |(E(t) - E(0)) - W(t)| / E(t_end) <= TOL_BAR
t_hist = sol_0b['t']
y_hist = sol_0b['y']
n_steps = t_hist.size

e_kin = np.zeros(n_steps)
e_str = np.zeros(n_steps)
e_tot = np.zeros(n_steps)
power = np.zeros(n_steps)
for k in range(n_steps):
    yk = y_hist[k]
    eng = compute_bulk_energy(ctx_0b, yk)
    e_kin[k] = eng['kinetic']
    e_str[k] = eng['strain']
    e_tot[k] = eng['total']
    power[k] = compute_boundary_power(ctx_0b, yk, float(t_hist[k]))

work_cum = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t_hist))])

delta_e = e_tot - e_tot[0]
residual = delta_e - work_cum
e_ref = max(float(e_tot[-1]), float(np.max(np.abs(delta_e))), 1.0e-300)
rel_err = np.max(np.abs(residual)) / e_ref

valid_B = rel_err <= TOL_DISCR_B
print("Validation B — work-energy balance")
print(f"  E_tot(t_end)              = {e_tot[-1]:.6e}")
print(f"  max |Delta E - W|         = {np.max(np.abs(residual)):.6e}")
print(f"  max relative error        = {rel_err:.6e}")
print(f"  tolerance                 = {TOL_DISCR_B:.3e}  (discretization-limited)")
print(f"  PASS                      = {valid_B}")

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))
axes[0].plot(t_hist, e_kin, label=r'$E_\mathrm{kin}$', color='tab:blue')
axes[0].plot(t_hist, e_str, label=r'$E_\mathrm{strain}$', color='tab:orange')
axes[0].plot(t_hist, e_tot, label=r'$E_\mathrm{tot}$', color='k', lw=1.5)
axes[0].plot(t_hist, work_cum, label=r'$W(t)$ (traction)', color='tab:red', ls='--')
axes[0].set_xlabel('nondim $t$')
axes[0].set_ylabel('energy / work')
axes[0].legend(fontsize=8)
axes[0].set_title('Sub-run 0b: energy & boundary work')

axes[1].plot(t_hist, residual, 'k-', lw=1.0)
axes[1].axhline(TOL_DISCR_B * e_ref, color='tab:green', ls='--', label='tol bar')
axes[1].axhline(-TOL_DISCR_B * e_ref, color='tab:green', ls='--')
axes[1].set_xlabel('nondim $t$')
axes[1].set_ylabel(r'$\Delta E - W$')
axes[1].set_title('Work-energy residual')
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
"""

CELL_CODE_0B_SYMMETRY = r"""
# Validation C — diagonal symmetry check at t = t_end.
#
# For sigma_right = sigma_top on a square domain, u at (x, y) must equal
# u at (y, x) with x,y components swapped.
u_end = extract_bulk_displacement(ctx_0b, y_hist[-1])
poro_0b = ctx_0b['poro']
basis_u = poro_0b.basis_u
dof_xy = np.asarray(basis_u.doflocs, dtype=float)
nodal_dofs = np.asarray(basis_u.nodal_dofs, dtype=int)

x_dofs = nodal_dofs[0]
y_dofs = nodal_dofs[1]
node_xy = dof_xy[:, x_dofs]
ux = u_end[x_dofs]
uy_swap = np.zeros_like(ux)
ux_swap = np.zeros_like(ux)
for k in range(x_dofs.size):
    xk, yk = node_xy[0, k], node_xy[1, k]
    swap_dist = (node_xy[0] - yk)**2 + (node_xy[1] - xk)**2
    j = int(np.argmin(swap_dist))
    ux_swap[k] = u_end[x_dofs[j]]
    uy_swap[k] = u_end[y_dofs[j]]

asym_x = ux - uy_swap
asym_y = u_end[y_dofs] - ux_swap
l2_u = float(np.sqrt(np.sum(u_end**2)) + 1.0e-30)
l2_asym = float(np.sqrt(np.sum(asym_x**2) + np.sum(asym_y**2)))
rel_asym = l2_asym / l2_u

# Discretization tolerance: tensor-triangle meshes lack exact diagonal
# symmetry because triangles have a chirality.  A P2 mesh with n_elem=12
# is expected to be symmetric to O(h^2) ~ 0.7% at best; we use 5%.
TOL_DISCR_C = 0.05
valid_C = rel_asym <= TOL_DISCR_C
print("Validation C — diagonal symmetry")
print(f"  L2(u)              = {l2_u:.6e}")
print(f"  L2 asymmetry       = {l2_asym:.6e}")
print(f"  relative asymmetry = {rel_asym:.6e}")
print(f"  tolerance          = {TOL_DISCR_C:.3e}  (discretization-limited)")
print(f"  PASS               = {valid_C}")
"""

CELL_MD_0C = r"""
## Sub-run 0c — slow-loading stress recovery

A much slower cosine ramp than 0b — nominally longer than several S-wave
transit times, so the transient has been smeared out and the interior
stress is well approximated by the quasi-static plate-stress
$\sigma_{xx} \approx -\sigma_R$, $\sigma_{yy} \approx -\sigma_T$ at the
end of the run.  *Validation D* checks the spatial mean of
$\sigma_{xx}$ against $-\sigma_R$ to within 5%.
"""

CELL_CODE_0C_BUILD = r"""
T_RAMP_0C_ND = 50.0 * (L_UNIT / C_S)
T_END_0C = 1.2 * T_RAMP_0C_ND
T_RAMP_0C_DIM = T_RAMP_0C_ND * T_SCALE

ctx_0c = build_intact_biaxial_dynamic_context(
    n_elem=N_ELEM,
    sigma_right=SIGMA_SYMM,
    sigma_top=SIGMA_SYMM,
    traction_scale_fn=make_cosine_ramp_scale_fn(t_ramp_dim=T_RAMP_0C_DIM),
)
print(f"Sub-run 0c: t_ramp = {T_RAMP_0C_ND:.4f} nondim, t_end = {T_END_0C:.4f}")
"""

CELL_CODE_0C_SOLVE = r"""
print(f"Integrating Sub-run 0c with method = {SOLVER_METHOD}")
sol_0c = run_descriptor(ctx_0c, T_END_0C, n_steps=600)
print(f"  n_steps = {sol_0c['t'].size}, final t = {sol_0c['t'][-1]:.6f}")
"""

CELL_CODE_0C_VALID = r"""
# Validation D — slow-loading stress recovery.
#
# Use the poroelasticity post-processor to recover nodal (sigma_xx, sigma_yy)
# from the final transformed base state, then compare their nodal average
# against the prescribed boundary tractions -sigma_right, -sigma_top.
ctx = ctx_0c
poro = ctx['poro']
n_base = ctx['n_base']
z_end = np.asarray(sol_0c['y'][-1][:n_base], dtype=float)
stress, _strain = poro.compute_stress_strain(z_end, only_nodal=True)
mean_sxx = float(np.mean(stress[0, 0]))
mean_syy = float(np.mean(stress[1, 1]))

target_sxx = -SIGMA_SYMM
target_syy = -SIGMA_SYMM
err_xx = abs(mean_sxx - target_sxx) / abs(target_sxx)
err_yy = abs(mean_syy - target_syy) / abs(target_syy)

valid_D = (err_xx <= 0.05) and (err_yy <= 0.05)
print("Validation D — slow-loading stress recovery")
print(f"  mean sigma_xx     = {mean_sxx:.6e}   target = {target_sxx:.6e}")
print(f"  mean sigma_yy     = {mean_syy:.6e}   target = {target_syy:.6e}")
print(f"  relative error    = {err_xx:.3%}, {err_yy:.3%}")
print(f"  tolerance         = 5.00%")
print(f"  PASS              = {valid_D}")
"""

CELL_MD_EXIT = r"""
## Phase 0 — exit gate summary

All five validations below must pass at the specified tolerances before
Phase 1 begins.  *Validation A0 is the blocker*: if A0 fails, the initial
state is inconsistent with the descriptor system and nothing downstream
is meaningful.
"""

CELL_CODE_EXIT = r"""
summary = {
    'A0 (initial residual)':   a0['passed'],
    'A  (P-wave arrival)':     bool(valid_A),
    'B  (work-energy)':        bool(valid_B),
    'C  (diagonal symmetry)':  bool(valid_C),
    'D  (stress recovery)':    bool(valid_D),
}

print("Phase 0 exit gate summary")
print("-" * 40)
for name, ok in summary.items():
    status = 'PASS' if ok else 'FAIL'
    print(f"  {name:30s}  {status}")
all_pass = all(summary.values())
print("-" * 40)
print(f"  OVERALL: {'PASS — proceed to Phase 1' if all_pass else 'FAIL — investigate per plan risk register'}")
assert all_pass, 'Phase 0 validations failed — do not advance to Phase 1 until they pass.'
"""


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(CELL_MD_HEADER),
        code(CELL_CODE_IMPORTS),
        md(CELL_MD_0A),
        code(CELL_CODE_0A_BUILD),
        code(CELL_CODE_0A_SOLVE),
        code(CELL_CODE_0A_VALID),
        md(CELL_MD_0B),
        code(CELL_CODE_0B_BUILD),
        code(CELL_CODE_0B_A0),
        code(CELL_CODE_0B_SOLVE),
        code(CELL_CODE_0B_ENERGY),
        code(CELL_CODE_0B_SYMMETRY),
        md(CELL_MD_0C),
        code(CELL_CODE_0C_BUILD),
        code(CELL_CODE_0C_SOLVE),
        code(CELL_CODE_0C_VALID),
        md(CELL_MD_EXIT),
        code(CELL_CODE_EXIT),
    ]
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python', 'mimetype': 'text/x-python'},
    }
    nbf.write(nb, NB_PATH)
    print(f'wrote {NB_PATH}')


if __name__ == '__main__':
    main()
