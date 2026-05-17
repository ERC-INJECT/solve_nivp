"""Smoke-test the new injection section.

Phase 1 (cheap):  build mesh + solver + augmented system, then verify
                  rhs_dyn_inj_eff and rhs_jac_dyn_inj_eff evaluate without
                  exception at y0 with a non-trivial source contribution.

Phase 2 (optional, slow): run a tiny 0.1-s time integration via scipy direct
                  solver; controlled by env SMOKE_INTEGRATE=1.

Run:
    python _inj_smoke_test.py             # phase 1 only
    SMOKE_INTEGRATE=1 python _inj_smoke_test.py   # phase 1 + 2
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

NB = '/home/david/Documents/Solve_ivp_ns/examples/embedded_crack_mohr_coulomb_ncp.ipynb'

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLBACKEND', 'Agg')

with open(NB) as f:
    nb = json.load(f)

ids_to_cells = {c.get('id'): c for c in nb['cells']}

# Cells to run for the build sequence
RUN_IDS_BUILD = [
    '2f218317',           # imports
    'wave_clock_override',
    'e119fa84',           # config + helpers
    '3d328595',           # mohr arithmetic
    '33cd134f',           # locked builder (for `mohr_tractions`, helpers)
    'inj_params', 'inj_build', 'inj_aug',
]

g = {'__name__': '__main__'}

def run_cell(cell_id, extra_after=''):
    cell = ids_to_cells[cell_id]
    src = ''.join(cell['source']) + extra_after
    print(f'\n--- cell {cell_id} ({len(src)} chars) ---', flush=True)
    t0 = time.time()
    exec(compile(src, f'<cell {cell_id}>', 'exec'), g)
    print(f'    [{time.time()-t0:.2f}s]', flush=True)

def main():
    t_total = time.time()
    for cid in RUN_IDS_BUILD:
        run_cell(cid)

    print('\n=== Phase 1: rhs / jac evaluation ===', flush=True)
    rhs_eff = g['rhs_dyn_inj_eff']
    jac_eff = g['rhs_jac_dyn_inj_eff']
    A_dyn   = g['A_dyn_inj']
    Bp      = g['Bp_inj']
    Q_scale = g['Q_scale_inj']
    Np      = g['Np_inj']
    q_fn    = g['q_inj_phys']

    n = A_dyn.shape[0]
    y0 = np.zeros(n)

    # t = 0: tanh(0) = 0 -> source contribution = 0
    f0 = rhs_eff(0.0, y0)
    f0_p = f0[:Np]
    print(f'rhs(t=0, y0): max |f|={np.max(np.abs(f0)):.3e}, max |f_p|={np.max(np.abs(f0_p)):.3e}')
    if np.max(np.abs(f0_p)) > 1e-10:
        print('  WARN: pressure-row residual nonzero at t=0 with q(0)=0')

    # t > 0: tanh(t/tau_ramp) > 0 -> source nonzero, pressure-row entries should pick up Bp*q/Q
    t_test = 5.0 / g['T_scale']  # 5 physical seconds
    f_test = rhs_eff(t_test, y0)
    f_test_p = f_test[:Np]
    expected = (Bp @ q_fn(t_test)) / Q_scale
    err = np.max(np.abs(f_test_p - expected.ravel()))
    print(f'rhs(t=5s, y0): max |f_p|={np.max(np.abs(f_test_p)):.3e}, '
          f'expected_max={np.max(np.abs(expected)):.3e}, '
          f'|f_p - Bp q/Q|_max={err:.3e}')
    assert err < 1e-12, "source contribution does not match Bp @ (q/Q)"
    print('  source wiring OK')

    J_test = jac_eff(t_test, y0)
    print(f'jac(t=5s, y0): shape={J_test.shape}, nnz={J_test.nnz}, '
          f'norm={float(np.sqrt((J_test.power(2)).sum())):.3e}')

    print(f'\nPhase 1 PASSED in {time.time()-t_total:.1f}s')

    if os.environ.get('SMOKE_INTEGRATE') != '1':
        print('Set SMOKE_INTEGRATE=1 to also run a 0.1-s integration.')
        return

    print('\n=== Phase 2: tiny integration via notebook inj_solve cell (PETSc/MUMPS) ===', flush=True)
    # Run the real inj_solve cell with TMAX cut to 1 physical second so the
    # PETSc/MUMPS Newton path is exercised but the run finishes in seconds.
    extra_pre = (
        '\nTMAX_PHYS_INJ = 1.0\nTMAX_INJ = TMAX_PHYS_INJ / T_scale\n'
        'ADAPTIVE_H_MAX_INJ = TMAX_INJ / 4.0\n'
        'print(f"[smoke] tmax_phys={TMAX_PHYS_INJ}, tmax_d={TMAX_INJ:.3g}, PETSc/MUMPS")\n'
    )
    exec(compile(extra_pre, '<phase2-pre>', 'exec'), g)
    t0 = time.time()
    run_cell('inj_solve')
    print(f'\nPhase 2 PASSED in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
