"""Static diagnostic: run the embedded-crack sliding case once with ALPHA_BIOT=0.2
and dump pore-pressure / displacement snapshots so we can inspect symmetry.

Usage:
    python _poro_field_diag.py [BOX_SCALE]

Outputs go to /tmp/poro_diag/.
"""
from __future__ import annotations

import os
import sys
import json
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['MPLBACKEND'] = 'Agg'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skfem.models.elasticity import lame_parameters

from poroelasticity import CGPoroelastostatics, CrackMeshBuilder
import solve_nivp
from solve_nivp.projected_radau_contact import build_projected_radau_contact


# ── Configuration (mirrors the notebook) ──────────────────────────────────
NU          = 0.25
G_SHEAR     = 22.0e3                # MPa
E_YOUNG     = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT  = 0.2
BETA_FLUID  = 8.5e-5
ETA_FLUID   = 2.0e-18 / 3600.0
K_PERM      = 1.0e-15 * 1.0e-6
DENSITY     = 1.05
BULK_MU_V   = 1.0e-1
BULK_LAM_V  = 1.0e-1
SCALE_L     = 1.0
SCALE_EPS   = 1.0e-3

XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0
N_ELEM_BASE = 20

CRACK_LENGTH = 0.4

SIGMA_RIGHT = 10.0
SIGMA_TOP   = 30.0

MU = 0.3

TMAX     = 5.0
H_FIXED  = 0.01

THETA_SLIDING = np.radians(27.0)
SNAPSHOT_TIMES = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]


def mohr_tractions(theta, S1, S3):
    sigma_N = (S1 + S3) / 2 + (S1 - S3) / 2 * np.cos(2 * theta)
    tau     = (S1 - S3) / 2 * np.sin(2 * theta)
    return sigma_N, tau


def run_sliding(
    box_scale: float,
    outdir: str,
    *,
    tmax: float = TMAX,
    T_robin: float | None = None,
) -> dict:
    """Run the sliding test for a given BOX_SCALE; save pore-pressure snapshots.

    Parameters
    ----------
    T_robin : float or None
        Crack interface Robin transmissivity. None keeps the library default
        (T=1e2, leaky). Pass 0.0 for an impermeable (sealed) crack.
    """

    os.makedirs(outdir, exist_ok=True)

    xmax = XMAX * box_scale
    ymax = YMAX * box_scale
    n_elem = max(int(round(N_ELEM_BASE * box_scale)), N_ELEM_BASE)
    crack_x0 = 0.5 * (XMIN + xmax)
    crack_y0 = 0.5 * (YMIN + ymax)

    LAM, MU_LAME = lame_parameters(E_YOUNG, NU)
    PARAMS = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)
    Mmod = LAM + 2 * MU_LAME
    Sigma_scale = Mmod * SCALE_EPS

    S1 = max(SIGMA_RIGHT, SIGMA_TOP)
    S3 = min(SIGMA_RIGHT, SIGMA_TOP)
    crack_theta_sliding = np.pi / 2 - THETA_SLIDING

    tag = f"box={box_scale},T={T_robin if T_robin is not None else 'default'}"
    print(f"[{tag}] domain [{XMIN},{xmax}] x [{YMIN},{ymax}], N_ELEM={n_elem}")
    print(f"[{tag}] crack centre=({crack_x0},{crack_y0}), length={CRACK_LENGTH}")

    builder = CrackMeshBuilder(
        XMIN, xmax, YMIN, ymax, n_elem,
        crack_theta=crack_theta_sliding,
        crack_x0=crack_x0, crack_y0=crack_y0,
        crack_length=CRACK_LENGTH,
        element_type='tri', conforming=True,
        verbose=False,
    )
    mesh, el_p, el_u, crack, h_mesh = builder.build()

    bc = {
        'v1': {'left': 0.0},
        'v2': {'bottom': 0.0},
        'dp_rate': {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0},
    }

    model_params = None
    if T_robin is not None:
        model_params = {'T': float(T_robin), 'tangential_flow': False}

    poro = CGPoroelastostatics(
        mesh=mesh,
        element_p=el_p,
        element_u=el_u,
        params=PARAMS,
        crack=crack,
        model_params=model_params,
        intorder=6,
        scales=(SCALE_L, SCALE_EPS),
        bc=bc,
        P_scale=None,
        verbose=False,
        free_memory=False,
        enforcement_type='nodal',
        apply_transform=True,
        include_taylor=False,
        include_hessian=False,
        include_sbm=False,
        density=DENSITY,
        crack_law='nonsmooth',
        rotate_crack_to_nt=True,
        bulk_viscosity={'mu_v': BULK_MU_V, 'lam_v': BULK_LAM_V},
    )

    poro.strip_multiplier_dynamics()
    projection, meta = poro.build_projection()
    dyn = poro.build_first_order_dynamic_system(meta, density=DENSITY)

    A_dyn = dyn['A']
    rhs_dyn = dyn['rhs']
    rhs_jac_dyn = dyn['rhs_jac']
    n_base = dyn['n_base']
    comp_slices = dyn['component_slices']

    Np = poro.basis_p.N
    Nu = poro.basis_u.N

    # ── Contact wiring (copied from notebook cell 18, sliding branch) ──
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

    T_u_mat = dyn['T_u']
    u_idx = dyn['u_state_indices']
    jmpu_local = np.searchsorted(u_idx, jmpu_all)
    assert np.all(u_idx[jmpu_local] == jmpu_all)
    D = T_u_mat[jmpu_local, :].tocsr()

    R_v = dyn.get('R_v')
    A_csr = poro.A.tocsr()
    B_u = (A_csr[Np:Np + Nu, :][:, jmpls_cols]).tocsr()
    if R_v is not None:
        B_u = (R_v @ B_u).tocsr()

    dirichlet_local = np.array(
        [d - Np for d in np.asarray(getattr(poro, '_dirichlet_dof_set', []), dtype=int)
         if Np <= d < Np + Nu],
        dtype=int,
    )
    if dirichlet_local.size:
        B_u = B_u.tolil()
        B_u[dirichlet_local, :] = 0.0
        B_u = B_u.tocsr()

    perm = [idx for k in range(n_c) for idx in (k, n_c + k)]
    B_u_perm = B_u[:, perm].tocsr()

    contacts = [
        {'vel_normal_idx': k, 'vel_tangential_idx': [n_c + k], 'mu': MU, 'e': 0.0}
        for k in range(n_c)
    ]

    n_phys = A_dyn.shape[0]
    n_extract = 2 * n_c
    gap_extract_dyn = sp.csr_matrix(
        (np.ones(n_extract), (np.arange(n_extract), jmpu_all)),
        shape=(n_extract, n_phys),
    )
    vel_extract_dyn = sp.hstack(
        [sp.csr_matrix((n_extract, n_base)), D], format='csr',
    )
    B_c_dyn = sp.vstack(
        [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm], format='csr',
    )
    B_n_dyn = B_c_dyn[:, 0::2]

    off_avgp = info['off_avgp']
    n_intf_p = info['n_intf_p']
    assert n_intf_p == n_c
    p_gamma_extract_dyn = sp.csr_matrix(
        (np.ones(n_c), (np.arange(n_c), off_avgp + np.arange(n_c))),
        shape=(n_c, n_phys),
    )
    pressure_normal_correction = -ALPHA_BIOT * (B_n_dyn @ p_gamma_extract_dyn)
    pressure_normal_correction.eliminate_zeros()

    def rhs_dyn_eff(t, y, *extra):
        return rhs_dyn(t, y, *extra) + pressure_normal_correction @ y

    def rhs_jac_dyn_eff(t, y, *extra):
        return rhs_jac_dyn(t, y, *extra) + pressure_normal_correction

    sigma_N_sliding, tau_sliding_val = mohr_tractions(THETA_SLIDING, S1, S3)
    s0_val = np.full(n_c, sigma_N_sliding / Sigma_scale)
    w0_val = np.full(n_c, tau_sliding_val / Sigma_scale)

    def get_s0_sliding(_y):
        return s0_val

    def get_w0_sliding(_y, k):
        return np.array([w0_val[int(k)]])

    flux_constraint = meta['constraints'][0]
    n_lam_s_dim = int(lam_s_sl.stop - lam_s_sl.start)
    zero_lam_s = {
        'g': lambda zf, *_a, _n=n_lam_s_dim: np.zeros(_n),
        'dg_dy': lambda zf, *_a, _n=n_lam_s_dim: np.zeros((_n, _n)),
        'y_slice': lam_s_sl,
        'q_slice': lam_s_sl,
    }

    cs = build_projected_radau_contact(
        A_dyn, rhs_dyn_eff, np.zeros(A_dyn.shape[0]),
        contacts=contacts,
        C_extract=gap_extract_dyn,
        D_extract=vel_extract_dyn,
        B=B_c_dyn,
        constraints=[flux_constraint, zero_lam_s],
        component_slices=comp_slices,
        rhs_jac=rhs_jac_dyn_eff,
        get_s0=get_s0_sliding,
        get_w0=get_w0_sliding,
        normal_r='auto',
        friction_r='auto',
        endpoint_inactive_handling='natural_map',
        reported_reaction_units='force',
    )

    # ── Solve ────────────────────────────────────────────────────────────
    n_aug = cs.y0.size
    nl_atol = np.full(n_aug, 1e-8)
    nl_rtol = np.full(n_aug, 1e-6)
    nl_atol[n_phys:] = 1e-10
    nl_rtol[n_phys:] = 0.0

    solver_opts = dict(cs.solver_opts)
    solver_opts.update(
        tol=1e-8, max_iter=500, rhs_jac=cs.rhs_jac,
        linear_solver='petsc',
        petsc_options={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        },
    )
    integrator_opts = dict(cs.integrator_opts)
    integrator_opts.update({'stages': 2, 'use_coupled_newton': True})

    print(f"[{tag}] solving sliding test, tmax={tmax}, n_phys={n_phys}, n_aug={n_aug}")
    t0 = time.time()
    (
        t_arr, y_arr, h_arr, fk_arr, info_arr, attempts
    ) = solve_nivp.solve_ivp_ns(
        fun=cs.rhs,
        t_span=(0, tmax),
        y0=cs.y0.copy(),
        method='radau_iia',
        integrator_opts=integrator_opts,
        projection=cs.projection,
        solver='semismooth_newton',
        projection_opts={'component_slices': cs.component_slices},
        solver_opts=solver_opts,
        adaptive=True,
        h0=H_FIXED,
        rtol=1e-3, atol=1e-5,
        skip_error_indices=[0],
        active_set_filter=False,
        return_attempts=True,
        adaptive_opts=dict(h0=H_FIXED, h_min=1e-5, h_max=tmax/2,
                           h_up=2.0, h_down=0.5, method_order=1),
        nl_atol=nl_atol, nl_rtol=nl_rtol,
        component_slices=cs.component_slices,
        verbose=False,
        A=cs.A,
        dae_var_weight='auto',
    )
    elapsed = time.time() - t0
    print(f"[{tag}] solve done in {elapsed:.1f}s, n_states={len(t_arr)}, "
          f"final t={t_arr[-1]:.3f}")

    # ── Build dimensional fields ────────────────────────────────────────
    z_arr = y_arr[:, :n_base]
    fields = poro.sol_to_fields_dict_with_dimensions(
        np.asarray(t_arr), z_arr.T,
        use_taylor_correction=False, constrained_l2=False,
    )

    p_field = np.asarray(fields['p'])      # (Np_p_basis, n_t)
    t_dim = np.asarray(fields['t'])
    u_field = np.asarray(fields['u'])      # (dim, Np_u, n_t)

    # Pressure-basis nodal coordinates
    p_dofs = poro.basis_p.doflocs           # shape (dim, Np_p)

    # ── Slip & cone diagnostics (mirror notebook) ──────────────────────
    slip_arr = np.asarray(y_arr[:, jmpu_t_idx], dtype=float)
    gap_arr = np.asarray(y_arr[:, jmpu_n_idx], dtype=float)
    r_pert = y_arr[:, n_phys:n_phys + 2 * n_c]
    r_n = (r_pert[:, 0::2] + s0_val[None, :]) * Sigma_scale
    r_t = (r_pert[:, 1::2] + w0_val[None, :]) * Sigma_scale
    cone_margin = MU * r_n - np.abs(r_t)

    # Pressure trace at the crack
    p_gamma_arr = y_arr[:, off_avgp:off_avgp + n_intf_p]   # nondim trace

    # ── Save snapshots and asymmetry diagnostics ───────────────────────
    diag = {
        'box_scale': float(box_scale),
        'T_robin': float(T_robin) if T_robin is not None else None,
        'T_robin_d': float(getattr(poro, 'T_robin_d', 0.0)),
        'has_tangential_flow': bool(poro.has_tangential_flow),
        'n_elem': int(n_elem),
        'h_mesh': float(h_mesh),
        'a_over_L': float(CRACK_LENGTH / 2.0 / max(xmax - XMIN, ymax - YMIN, 1e-30)),
        'tmax': float(tmax),
        'n_states': int(len(t_arr)),
        'elapsed_s': float(elapsed),
        'crack_centre': [float(crack_x0), float(crack_y0)],
        'crack_theta_sliding_deg': float(np.degrees(crack_theta_sliding)),
        'sigma_N_MPa': float(sigma_N_sliding),
        'tau_MPa': float(tau_sliding_val),
        'max_abs_slip_nondim': float(np.max(np.abs(slip_arr))),
        'max_abs_slip_phys': float(np.max(np.abs(slip_arr)) * SCALE_L * SCALE_EPS),
        'max_abs_p_MPa': float(np.max(np.abs(p_field))),
        'max_abs_p_gamma_MPa': float(np.max(np.abs(p_gamma_arr)) * Sigma_scale),
        'min_cone_margin_MPa': float(np.min(cone_margin)),
        'final_max_abs_p_MPa': float(np.max(np.abs(p_field[:, -1]))),
        'mean_p_final_MPa': float(np.mean(p_field[:, -1])),
        'l2_p_final_MPa': float(np.sqrt(np.mean(p_field[:, -1] ** 2))),
    }

    # Snapshot indices closest to requested times. t_dim may be scaled by the
    # poroelasticity time scale T = L^2/C; fall back to evenly-spaced indices
    # if none of the requested times fit.
    snap_idxs = []
    t_final = t_dim[-1]
    for ts in SNAPSHOT_TIMES:
        if ts <= t_final:
            snap_idxs.append(int(np.argmin(np.abs(t_dim - ts))))
        else:
            scaled = ts / SNAPSHOT_TIMES[-1] * t_final
            snap_idxs.append(int(np.argmin(np.abs(t_dim - scaled))))
    snap_idxs = sorted(set(snap_idxs))
    if not snap_idxs:
        snap_idxs = list(np.linspace(0, len(t_dim) - 1, 6).astype(int))
        snap_idxs = sorted(set(snap_idxs))
    print(f"[{tag}] t_dim range = [{t_dim[0]:.3e}, {t_dim[-1]:.3e}], "
          f"snapshots at {[float(t_dim[i]) for i in snap_idxs]}")
    diag['snapshot_indices'] = snap_idxs
    diag['snapshot_times'] = [float(t_dim[i]) for i in snap_idxs]

    # Antisymmetry probe: p(x,y) vs p reflected through crack centre.
    # Build an interpolator at (2*xc - x, 2*yc - y) and compare.
    from scipy.interpolate import LinearNDInterpolator
    px = p_dofs[0]
    py = p_dofs[1]
    asym = []
    for i in snap_idxs:
        pi = p_field[:, i]
        interp = LinearNDInterpolator(np.c_[px, py], pi, fill_value=np.nan)
        p_reflected = interp(2 * crack_x0 - px, 2 * crack_y0 - py)
        mask = np.isfinite(p_reflected)
        if not np.any(mask):
            asym.append(None)
            continue
        # p should be antisymmetric → p(reflected) ≈ -p(original)
        anti_resid = pi[mask] + p_reflected[mask]
        sym_resid  = pi[mask] - p_reflected[mask]
        denom = max(np.linalg.norm(pi[mask]), 1e-30)
        asym.append({
            't': float(t_dim[i]),
            'antisym_relerr_180deg': float(np.linalg.norm(anti_resid) / denom),
            'sym_relerr_180deg':     float(np.linalg.norm(sym_resid)  / denom),
            'fraction_unmasked': float(mask.mean()),
        })
    diag['antisymmetry_probe_180deg'] = asym

    # ── Plot snapshots: 2x3 grid of pressure fields ────────────────────
    n_plots = len(snap_idxs)
    cols = min(3, max(1, n_plots))
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows),
                             squeeze=False)
    pmax_global = float(np.max(np.abs(p_field[:, snap_idxs]))) if snap_idxs else 1.0
    pmax_global = max(pmax_global, 1e-30)
    for ax_i, idx in enumerate(snap_idxs):
        r, c = divmod(ax_i, cols)
        ax = axes[r][c]
        cf = ax.tricontourf(
            px, py, p_field[:, idx],
            levels=21, cmap='RdBu_r',
            vmin=-pmax_global, vmax=pmax_global,
        )
        # Crack line
        a = CRACK_LENGTH / 2
        cosT, sinT = np.cos(crack_theta_sliding), np.sin(crack_theta_sliding)
        # crack_theta is measured CCW from y-axis -> tangent direction:
        tx, ty = sinT, cosT
        ax.plot([crack_x0 - a*tx, crack_x0 + a*tx],
                [crack_y0 - a*ty, crack_y0 + a*ty],
                'k-', lw=2)
        ax.plot([crack_x0], [crack_y0], 'k+', ms=10)
        ax.set_xlim(XMIN, xmax)
        ax.set_ylim(YMIN, ymax)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_dim[idx]:.3f}s\nmax|p|={np.max(np.abs(p_field[:, idx])):.2e} MPa')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='p [MPa]')
    for k in range(n_plots, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis('off')
    fig.suptitle(
        f'Pore pressure field — sliding test (BOX_SCALE={box_scale}, '
        f'ALPHA_BIOT={ALPHA_BIOT}, a/L={diag["a_over_L"]:.3f})'
    )
    fig.tight_layout()
    out_png = os.path.join(outdir, f'p_field_box{box_scale}_T{T_robin if T_robin is not None else 'default'}.png')
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[{tag}] saved {out_png}")

    # Asymmetry residual map at final snapshot
    if snap_idxs:
        i = snap_idxs[-1]
        pi = p_field[:, i]
        interp = LinearNDInterpolator(np.c_[px, py], pi, fill_value=np.nan)
        p_reflected = interp(2 * crack_x0 - px, 2 * crack_y0 - py)
        anti_resid = pi + p_reflected   # NaN where interp fails
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.5))
        cf2 = ax2[0].tricontourf(px, py, pi, levels=21, cmap='RdBu_r',
                                 vmin=-pmax_global, vmax=pmax_global)
        ax2[0].set_title(f'p at t={t_dim[i]:.3f}s')
        ax2[0].set_aspect('equal')
        plt.colorbar(cf2, ax=ax2[0], fraction=0.046, pad=0.04, label='p [MPa]')

        # Plot residual where we have data
        mask = np.isfinite(anti_resid)
        if np.any(mask):
            r_max = max(np.max(np.abs(anti_resid[mask])), 1e-30)
            cf3 = ax2[1].tricontourf(
                px[mask], py[mask], anti_resid[mask],
                levels=21, cmap='PuOr', vmin=-r_max, vmax=r_max,
            )
            ax2[1].set_title(
                'p(x,y) + p(2c−x, 2c−y)\n'
                '(zero ⇔ perfect 180° dipole symmetry)'
            )
            ax2[1].set_aspect('equal')
            plt.colorbar(cf3, ax=ax2[1], fraction=0.046, pad=0.04,
                         label='residual [MPa]')
        for axx in ax2:
            axx.set_xlim(XMIN, xmax)
            axx.set_ylim(YMIN, ymax)
        fig2.tight_layout()
        out_png2 = os.path.join(outdir, f'p_antisym_box{box_scale}_T{T_robin if T_robin is not None else 'default'}.png')
        fig2.savefig(out_png2, dpi=130)
        plt.close(fig2)
        print(f"[{tag}] saved {out_png2}")

    # ── Slip profile + max p timeseries ────────────────────────────────
    fig3, ax3 = plt.subplots(2, 1, figsize=(8, 6))
    ax3[0].plot(t_dim, np.max(np.abs(slip_arr), axis=1) * SCALE_L * SCALE_EPS, 'r-')
    ax3[0].set_xlabel('t [s]')
    ax3[0].set_ylabel('max |slip| [phys units]')
    ax3[0].grid(True, alpha=0.3)
    ax3[0].set_title(f'Sliding test BOX_SCALE={box_scale}')
    ax3[1].plot(t_dim, np.max(np.abs(p_field), axis=0), 'b-', label='max |p| (interior)')
    ax3[1].plot(t_dim,
                np.max(np.abs(p_gamma_arr), axis=1) * Sigma_scale, 'g--',
                label='max |p| at crack')
    ax3[1].set_xlabel('t [s]')
    ax3[1].set_ylabel('p [MPa]')
    ax3[1].grid(True, alpha=0.3)
    ax3[1].legend()
    fig3.tight_layout()
    out_png3 = os.path.join(outdir, f'timeseries_box{box_scale}_T{T_robin if T_robin is not None else 'default'}.png')
    fig3.savefig(out_png3, dpi=130)
    plt.close(fig3)
    print(f"[{tag}] saved {out_png3}")

    # Persist trajectory data
    np.savez_compressed(
        os.path.join(outdir, f'data_box{box_scale}_T{T_robin if T_robin is not None else 'default'}.npz'),
        t=t_dim, p=p_field, p_dofs=p_dofs,
        slip=slip_arr, gap=gap_arr, r_n=r_n, r_t=r_t,
        cone_margin=cone_margin,
        crack_centre=np.array([crack_x0, crack_y0]),
        crack_theta=np.array([crack_theta_sliding]),
        h_mesh=np.array([h_mesh]), n_elem=np.array([n_elem]),
    )

    return diag


def main(argv):
    """Args: BOX_SCALE [T_ROBIN].  Runs one config per (box_scale, T_robin)."""
    if len(argv) >= 3:
        box_scales = [float(argv[1])]
        T_list = [None if argv[2] == 'default' else float(argv[2])]
    elif len(argv) == 2:
        box_scales = [float(argv[1])]
        T_list = [None]
    else:
        box_scales = [1.0]
        T_list = [None]

    outdir = '/tmp/poro_diag'
    os.makedirs(outdir, exist_ok=True)

    diags = []
    for bs in box_scales:
        for tr in T_list:
            try:
                d = run_sliding(bs, outdir, T_robin=tr)
                diags.append(d)
            except Exception as exc:
                print(f"[box={bs}, T={tr}] FAILED: {exc!r}")
                import traceback; traceback.print_exc()

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(diags, f, indent=2)
    print(f"summary saved to {os.path.join(outdir, 'summary.json')}")
    for d in diags:
        print(json.dumps(d, indent=2, default=str))


if __name__ == '__main__':
    main(sys.argv)
