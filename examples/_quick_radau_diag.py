"""Quick diagnostic: Radau vs BE at small cone violation."""
import os
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import scipy.sparse as sp
import warnings
from skfem.models.elasticity import lame_parameters
from poroelasticity import CGPoroelastostatics, CrackMeshBuilder
from solve_nivp.ncp_contact import build_dynamic_ncp_contact
import solve_nivp

NU = 0.25
G_SHEAR = 22.0e3
E_YOUNG = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT = 0.0
BETA_FLUID = 8.5e-5
ETA_FLUID = 2.0e-18 / 3600.0
K_PERM = 1.0e-15 * 1.0e-6
DENSITY = 1.0
SCALE_L = 1.0
SCALE_EPS = 1.0e-3

XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0
N_ELEM = 20
CRACK_X0, CRACK_Y0 = 0.5, 0.5
CRACK_LENGTH = 0.4

SIGMA_RIGHT = 10.0
SIGMA_TOP = 30.0
MU_FRIC = 0.3
S1 = max(SIGMA_RIGHT, SIGMA_TOP)
S3 = min(SIGMA_RIGHT, SIGMA_TOP)

theta_crit = 0.5 * np.arctan(1.0 / MU_FRIC)
LAM, MU_LAME = lame_parameters(E_YOUNG, NU)


def mohr_tractions(theta):
    sigma_N = (S1 + S3) / 2 + (S1 - S3) / 2 * np.cos(2 * theta)
    tau = (S1 - S3) / 2 * np.sin(2 * theta)
    return sigma_N, tau


def build_system(theta_mohr, bulk_v):
    crack_theta = np.pi / 2 - theta_mohr
    sigma_N, tau_val = mohr_tractions(theta_mohr)
    params = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)

    builder = CrackMeshBuilder(
        XMIN, XMAX, YMIN, YMAX, N_ELEM,
        crack_theta=crack_theta,
        crack_x0=CRACK_X0, crack_y0=CRACK_Y0,
        crack_length=CRACK_LENGTH,
        element_type='tri', conforming=True, verbose=False,
    )
    mesh, el_p, el_u, crack, h_mesh = builder.build()

    bc = {
        'v1': {'left': 0.0},
        'v2': {'bottom': 0.0},
        'dp_rate': {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0},
    }
    poro = CGPoroelastostatics(
        mesh=mesh, element_p=el_p, element_u=el_u, params=params,
        crack=crack, intorder=6, scales=(SCALE_L, SCALE_EPS), bc=bc,
        P_scale=None, verbose=False, free_memory=False,
        enforcement_type='nodal', apply_transform=True,
        include_taylor=False, include_hessian=False, include_sbm=False,
        density=DENSITY, crack_law='nonsmooth', rotate_crack_to_nt=True,
        bulk_viscosity={'mu_v': bulk_v, 'lam_v': bulk_v},
    )
    poro.strip_multiplier_dynamics()
    projection, meta = poro.build_projection()
    dyn = poro.build_first_order_dynamic_system(meta, density=DENSITY)

    A_dyn = dyn['A']
    n_base = dyn['n_base']
    comp_slices = dyn['component_slices']
    Np = poro.basis_p.N
    Nu = poro.basis_u.N

    info = poro._transform_info
    n_c = int(poro.n_lambda_q)
    lam_s_sl = meta['lam_s_sl']

    off_jmpu = info['off_jmpu']
    n_iu = info['n_intf_u']
    off_jmpls = info['off_jmpls']
    n_lsig = info['n_lam_sig']

    jmpu_n_idx = np.arange(off_jmpu, off_jmpu + n_c)
    jmpu_t_idx = np.arange(off_jmpu + n_c, off_jmpu + n_iu)
    jmpu_all = np.concatenate([jmpu_n_idx, jmpu_t_idx])
    jmpls_cols = np.arange(off_jmpls, off_jmpls + n_lsig)

    T_u = dyn['T_u']
    u_idx = dyn['u_state_indices']
    jmpu_local = np.searchsorted(u_idx, jmpu_all)
    D = T_u[jmpu_local, :].tocsr()

    R_v = dyn.get('R_v')
    A_csr = poro.A.tocsr()
    B_u = -(A_csr[Np:Np + Nu, :][:, jmpls_cols]).tocsr()
    if R_v is not None:
        B_u = (R_v @ B_u).tocsr()

    dirichlet_local = np.array(
        [d - Np for d in np.asarray(getattr(poro, '_dirichlet_dof_set', []), dtype=int)
         if Np <= d < Np + Nu], dtype=int,
    )
    if dirichlet_local.size:
        B_u = B_u.tolil()
        B_u[dirichlet_local, :] = 0.0
        B_u = B_u.tocsr()

    perm = [idx for k in range(n_c) for idx in (k, n_c + k)]
    B_u_perm = B_u[:, perm].tocsr()

    contacts = [
        {'vel_normal_idx': k, 'vel_tangential_idx': [n_c + k], 'mu': MU_FRIC, 'e': 0.0}
        for k in range(n_c)
    ]

    n_extract = 2 * n_c
    n_aug_phys = A_dyn.shape[0]

    gap_extract = sp.csr_matrix(
        (np.ones(n_extract), (np.arange(n_extract), jmpu_all)),
        shape=(n_extract, n_aug_phys),
    )
    vel_extract = sp.hstack(
        [sp.csr_matrix((n_extract, n_base)), D], format='csr',
    )
    B_dyn = sp.vstack(
        [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm], format='csr',
    )

    get_s0 = lambda y, _sN=sigma_N: _sN
    get_w0 = lambda y, k, _tau=tau_val: np.array([_tau])

    flux_constraint = meta['constraints'][0]
    n_lam_s_dim = int(lam_s_sl.stop - lam_s_sl.start)
    zero_lam_s = {
        'g': lambda zf, *_a, _n=n_lam_s_dim: np.zeros(_n),
        'dg_dy': lambda zf, *_a, _n=n_lam_s_dim: np.zeros((_n, _n)),
        'y_slice': lam_s_sl,
        'q_slice': lam_s_sl,
    }

    cs = build_dynamic_ncp_contact(
        A=A_dyn, rhs_smooth=dyn['rhs'], rhs_jac=dyn['rhs_jac'],
        y0=np.zeros(A_dyn.shape[0]),
        contacts=contacts, B=B_dyn,
        gap_extract=gap_extract, vel_extract=vel_extract,
        constraints=[flux_constraint, zero_lam_s],
        component_slices=comp_slices,
        offset_coupling_mode='constitutive_shift',
        get_s0=get_s0, get_w0=get_w0,
        ncp_type='fischer_burmeister',
        normal_r='auto', friction_r='auto',
        inactive_handling='ncp',
    )

    return cs, A_dyn, n_base


def solve_with(cs, A_dyn, n_base, method, h_fixed, tmax):
    n_aug = cs.y0.size
    n_phys = A_dyn.shape[0]

    nl_atol = np.full(n_aug, 1e-8)
    nl_rtol = np.full(n_aug, 1e-6)
    nl_atol[n_phys:] = 1e-10
    nl_rtol[n_phys:] = 0.0

    solver_opts = dict(cs.solver_opts)
    solver_opts.pop('cold_start_slices', None)
    solver_opts['damped_step_fraction'] = 1.0
    solver_opts['diagonal_regularization'] = 0.0
    solver_opts.update(tol=1e-8, max_iter=500, rhs_jac=cs.rhs_jac, linear_solver='splu')

    integrator_opts = dict(cs.integrator_opts)
    if method == 'radau_iia':
        integrator_opts.update({'stages': 2, 'use_coupled_newton': True})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        t, y, h, fk, info_out = solve_nivp.solve_ivp_ns(
            fun=cs.rhs,
            t_span=(0, tmax),
            y0=cs.y0.copy(),
            method=method,
            integrator_opts=integrator_opts,
            projection=cs.projection,
            solver='semismooth_newton',
            projection_opts={'rhok': 1.0, 'component_slices': cs.component_slices},
            solver_opts=solver_opts,
            adaptive=False, h0=h_fixed,
            nl_atol=nl_atol, nl_rtol=nl_rtol,
            component_slices=cs.component_slices,
            verbose=False,
            A=cs.A,
            dae_var_weight='auto',
            active_set_filter=False,
        )

    y_fin = y[-1]
    u_max = np.max(np.abs(y_fin[:n_base]))
    r_max = np.max(np.abs(y_fin[n_phys:]))

    return len(t), t[-1], u_max, r_max


if __name__ == '__main__':
    print(f"θ_crit = {np.degrees(theta_crit):.2f}°")
    print()

    # θ just barely above critical
    theta_test = np.radians(37.0)
    sN, tau = mohr_tractions(theta_test)
    ratio = abs(tau) / (MU_FRIC * sN)
    print(f"θ = 37.0°, ratio = {ratio:.4f}")
    print()

    configs = [
        # (bulk_v, method, h, tmax)
        (1e6, 'backward_euler', 0.01, 0.5),
        (1e6, 'backward_euler', 0.1, 2.0),
        (1e6, 'radau_iia', 0.1, 2.0),
        (1e4, 'backward_euler', 0.01, 0.5),
        (1e4, 'backward_euler', 0.1, 2.0),
        (1e4, 'radau_iia', 0.1, 2.0),
        (1e2, 'backward_euler', 0.01, 0.5),
        (1e2, 'backward_euler', 0.1, 2.0),
        (1e2, 'radau_iia', 0.1, 2.0),
        (1e2, 'radau_iia', 0.01, 0.5),
        (1e0, 'backward_euler', 0.01, 0.5),
        (1e0, 'radau_iia', 0.1, 2.0),
        (1e0, 'radau_iia', 0.01, 0.5),
    ]

    print(f"{'bulk_v':>10} {'method':>16} {'h':>6} {'tmax':>6} {'steps':>6} "
          f"{'t_final':>8} {'max|u|':>12} {'max|r|':>12} {'sliding?':>9}")
    print('-' * 100)

    for bulk_v, method, h, tmax in configs:
        try:
            cs, A_dyn, n_base = build_system(theta_test, bulk_v)
            nsteps, tfin, umax, rmax = solve_with(cs, A_dyn, n_base, method, h, tmax)
            sliding = 'YES' if umax > 1e-8 else 'no'
            print(f"{bulk_v:>10.0e} {method:>16} {h:>6.2f} {tmax:>6.1f} {nsteps:>6} "
                  f"{tfin:>8.4f} {umax:>12.3e} {rmax:>12.3e} {sliding:>9}")
        except Exception as e:
            print(f"{bulk_v:>10.0e} {method:>16} {h:>6.2f} {tmax:>6.1f}  ERROR: {e}")
