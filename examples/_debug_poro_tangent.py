"""Minimal poro test for reduced-space De Saxcé tangent_cone_split.
Reuses the poro setup but only runs 1 step to diagnose the tangent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import scipy.sparse as sp
import time

# Exec the poro setup (everything before the solve call)
_setup_code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '_run_poro_radau_desaxce_debug.py')).read()
# Truncate at the solve call
_cut = _setup_code.find("start = time.time()")
exec(compile(_setup_code[:_cut].replace("os.path.dirname(__file__)",
     f"'{os.path.dirname(os.path.abspath(__file__))}'"), "<poro_setup>", "exec"))

# Now build the REDUCED-SPACE backend
from solve_nivp.desaxce_contact import build_dynamic_desaxce_contact

print(f"\n{'='*60}")
print("Building reduced-space De Saxcé contact backend")
print(f"{'='*60}")
print(f"y0.size = {y0.size}")
print(f"ndofs_aug = {ndofs_aug}")
print(f"slip_idx = {slip_idx}")
print(f"vtn_idx = {vtn_idx}, vtt_idx = {vtt_idx}")
print(f"u1n_idx = {u1n_idx}, u2t_idx = {u2t_idx}")
print(f"TPloc_idx['right'] = {TPloc_idx['right']}")
print(f"contact_vel_C nonzero: {list(zip(*contact_vel_C.nonzero()))} = {contact_vel_C.data}")
print(f"contact_gap_C nonzero: {list(zip(*contact_gap_C.nonzero()))} = {contact_gap_C.data}")

cs_red = build_dynamic_desaxce_contact(
    A=E_D,
    rhs_smooth=rhs_plant,
    y0=y0,
    contacts=contact_blocks,
    gap_extract=contact_gap_C,
    vel_extract=contact_vel_C,
    B=contact_B,
    component_slices=contact_component_slices,
    gap_func=None,
    get_s0=contact_s0_force_vec,
    get_w0=contact_w0_force_vec,
    rhs_jac=rhs_jac_aug,
)

proj = cs_red.projection
print(f"proj.n_phys = {proj.n_phys}")
print(f"proj._n_phys = {getattr(proj, '_n_phys', 'NOT SET')}")
print(f"has _vectorize_contact_params: {hasattr(proj, '_vectorize_contact_params')}")

# Test: evaluate mu at y0 and at y0 with perturbed slip
vcp = proj._vectorize_contact_params
mu0_val, beta0_val = vcp(y0, t=0.0, Fk_val=None)
print(f"\nmu(y0) = {mu0_val}")
print(f"beta(y0) = {beta0_val}")

y_pert = y0.copy()
y_pert[slip_idx] += 1e-4
mu_pert, _ = vcp(y_pert, t=0.0, Fk_val=None)
print(f"mu(y0 + eps*e_slip) = {mu_pert}")
print(f"dmu/d(slip) ≈ {(mu_pert - mu0_val) / 1e-4}")

# But does vcp receive the full y0 (including slip)?
n_phys_check = proj._n_phys
print(f"\nn_phys = {n_phys_check}")
print(f"y_cur_phys would be y[:n_phys] = y[:{n_phys_check}]")
print(f"slip_idx = {slip_idx}, in range? {slip_idx < n_phys_check}")

# Simulate one projection call and check tangent_cone_split
h = 0.005
t_new = h
prev_state = y0.copy()

# Call project to populate _last_solve_data
candidate = y0.copy()  # just use y0 as candidate for now
proj_out = proj.project(
    y0, candidate, rhok=1.0, t=t_new, Fk_val=None,
    prev_state=prev_state, step_size=h,
)

data = proj._last_solve_data
print(f"\n_last_solve_data keys: {list(data.keys()) if data else 'None'}")
if data:
    print(f"  active_blocks: {data['active_blocks']}")
    print(f"  y_cur_phys.shape: {data['y_cur_phys'].shape}")
    print(f"  y_cur_phys[slip_idx]: {data['y_cur_phys'][slip_idx] if slip_idx < len(data['y_cur_phys']) else 'OUT OF RANGE'}")
    solved = data['solved']
    if solved:
        print(f"  reaction: {solved['reaction']}")
        print(f"  mu: {solved['mu']}")
        print(f"  alpha: {solved['alpha']}")

# Now call tangent_cone_split
t0 = time.time()
Dproj, Dstate = proj.tangent_cone_split(
    candidate, y0, rhok=1.0, t=t_new, Fk_val=None,
    prev_state=prev_state, step_size=h,
)
dt = time.time() - t0

print(f"\ntangent_cone_split took {dt:.4f}s")
print(f"Dproj nnz: {Dproj.nnz if sp.issparse(Dproj) else np.count_nonzero(Dproj)}")
print(f"Dstate nnz: {Dstate.nnz if sp.issparse(Dstate) else np.count_nonzero(Dstate)}")

if sp.issparse(Dstate):
    Dstate_d = Dstate.toarray()
else:
    Dstate_d = np.asarray(Dstate)

if np.any(np.abs(Dstate_d) > 1e-15):
    nz_rows, nz_cols = np.nonzero(np.abs(Dstate_d) > 1e-15)
    print(f"Non-zero Dstate entries: {len(nz_rows)}")
    print(f"Max |Dstate| = {np.max(np.abs(Dstate_d)):.6e}")
    print(f"Dstate col norms: {np.linalg.norm(Dstate_d, axis=0).max():.6e}")
    key_dofs = [vtn_idx, vtt_idx, slip_idx]
    for r in key_dofs:
        for c in key_dofs:
            if abs(Dstate_d[r, c]) > 1e-15:
                print(f"  Dstate[{r}, {c}] = {Dstate_d[r, c]:.6e}")
else:
    print("*** Dstate is all zeros! ***")

# Determinism check + y_free availability
print(f"\n=== Determinism and y_free check ===")
n_st = proj.n_phys
implicit_eq = lambda yy: E_D @ ((yy - y0) / h) - rhs_plant(t_new, yy)

F_y = implicit_eq(y0)
cand_0 = y0 - 1.0 * F_y

# Call project multiple times with identical inputs
P_calls = []
for trial in range(3):
    proj._last_reaction_full[:] = 0.0  # reset warm start
    P_i = proj.project(y0.copy(), cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                       prev_state=y0.copy(), step_size=h)
    P_calls.append(P_i.copy())
    inner_info = proj._last_inner_info
    print(f"  trial {trial}: P[vtt]={P_i[vtt_idx]:.12e}, "
          f"inner_iters={inner_info['iterations']}, "
          f"inner_res={inner_info['residual']:.3e}, "
          f"R=[{proj._last_reaction_full[0]:.6e}, {proj._last_reaction_full[1]:.6e}]")

max_variation = max(np.max(np.abs(P_calls[i] - P_calls[0])) for i in range(1, 3))
print(f"  Max variation across trials: {max_variation:.3e}")

P_0 = P_calls[0]
print(f"P(y0, cand0)[vtt] = {P_0[vtt_idx]:.10e}")

# Verify P = y_free + G @ R decomposition
proj._last_reaction_full[:] = 0.0
P_0 = proj.project(y0.copy(), cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                    prev_state=y0.copy(), step_size=h)
R_converged = proj._last_reaction_full.copy()

data = proj._last_solve_data
solved = data["solved"]
G_state = solved["G_state"]
R_active = solved["reaction"]
G_corr = G_state @ R_active

# Reconstruct what project() should return
# If y_free is available: P = y_free + G_corr
# Check by computing residual
print(f"\n=== P = y_free + G@R verification ===")
# Get y_free from the local model by calling model builder directly
from types import SimpleNamespace as NS
proj._last_reaction_full[:] = R_converged
result2 = proj._solve_full_reaction(y0.copy(), cand_0.copy(), t=t_new, Fk_val=F_y,
                                     prev_state=y0.copy(), step_size=h)
y_free_val = result2.get("y_free")
if y_free_val is not None:
    reconstructed = y_free_val + G_corr
    print(f"y_free[vtt] = {y_free_val[vtt_idx]:.10e}")
    print(f"G_corr[vtt] = {G_corr[vtt_idx]:.10e}")
    print(f"P[vtt]      = {P_0[vtt_idx]:.10e}")
    print(f"y_free+G@R  = {reconstructed[vtt_idx]:.10e}")
    print(f"|P - (y_free+G@R)| = {np.max(np.abs(P_0 - reconstructed)):.3e}")
    # Also check: does y_free depend on candidate?
    proj._last_reaction_full[:] = R_converged
    cand_test = cand_0.copy()
    cand_test[vtt_idx] += 1.0  # BIG perturbation
    result3 = proj._solve_full_reaction(y0.copy(), cand_test, t=t_new, Fk_val=F_y,
                                         prev_state=y0.copy(), step_size=h)
    y_free_test = result3.get("y_free")
    R_test = result3["solved"]["reaction"] if result3["solved"] else None
    if y_free_test is not None:
        print(f"\nWith cand[vtt] += 1.0:")
        print(f"  y_free[vtt] = {y_free_test[vtt_idx]:.10e} (was {y_free_val[vtt_idx]:.10e})")
        print(f"  |y_free_diff| = {np.max(np.abs(y_free_test - y_free_val)):.3e}")
    if R_test is not None:
        print(f"  R = {R_test} (was {R_active})")
        print(f"  |R_diff| = {np.max(np.abs(R_test - R_active)):.3e}")
else:
    print("y_free is None!")

# Direct instrumented test: what changes between base and state_only?
print(f"\n=== Instrumented state_only perturbation at vtt ===")
# Patch _local_model_builder to trace what's different
_orig_lmb = proj.local_model_builder
_lmb_calls = []
def _traced_lmb(**kw):
    result = _orig_lmb(**kw)
    _lmb_calls.append({
        "u_free": result["u_free"].copy(),
        "mu": result["mu"].copy(),
        "alpha": result["alpha"].copy(),
        "offset": result["offset"].copy() if result.get("offset") is not None else None,
        "warm_start": result["warm_start"].copy(),
        "y_free": result["y_free"].copy() if result.get("y_free") is not None else None,
        "current_state_vtt": float(kw.get("current_state", np.zeros(1))[vtt_idx]),
    })
    return result
proj.local_model_builder = _traced_lmb

# Base call
_lmb_calls.clear()
proj._last_reaction_full[:] = R_converged
P_base = proj.project(y0.copy(), cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                       prev_state=y0.copy(), step_size=h)
base_model = _lmb_calls[-1]
R_base_full = proj._last_reaction_full.copy()

# State-only perturbation at vtt
eps_j = 1e-7
y_p = y0.copy()
y_p[vtt_idx] += eps_j
_lmb_calls.clear()
proj._last_reaction_full[:] = R_converged
P_pert = proj.project(y_p, cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                       prev_state=y0.copy(), step_size=h)
pert_model = _lmb_calls[-1]
R_pert_full = proj._last_reaction_full.copy()

print(f"  Base current_state[vtt] = {base_model['current_state_vtt']}")
print(f"  Pert current_state[vtt] = {pert_model['current_state_vtt']}")
print(f"  |u_free diff|  = {np.max(np.abs(pert_model['u_free'] - base_model['u_free'])):.3e}")
print(f"  |mu diff|      = {np.max(np.abs(pert_model['mu'] - base_model['mu'])):.3e}")
print(f"  |alpha diff|   = {np.max(np.abs(pert_model['alpha'] - base_model['alpha'])):.3e}")
if base_model["offset"] is not None:
    print(f"  |offset diff|  = {np.max(np.abs(pert_model['offset'] - base_model['offset'])):.3e}")
print(f"  |warm_start diff| = {np.max(np.abs(pert_model['warm_start'] - base_model['warm_start'])):.3e}")
if base_model["y_free"] is not None:
    print(f"  |y_free diff|  = {np.max(np.abs(pert_model['y_free'] - base_model['y_free'])):.3e}")
print(f"  |R diff|       = {np.max(np.abs(R_pert_full - R_base_full)):.3e}")
dP = (P_pert - P_base) / eps_j
print(f"  dP[vtt]        = {dP[vtt_idx]:.6e}")
print(f"  |dP|_inf       = {np.max(np.abs(dP)):.3e}")

proj.local_model_builder = _orig_lmb

# Warm-start consistency check: does P_0 depend on warm start?
print(f"\n=== Warm-start consistency ===")
proj._last_reaction_full[:] = R_converged
P_0_warm = proj.project(y0.copy(), cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                         prev_state=y0.copy(), step_size=h)
R_warm = proj._last_reaction_full.copy()
print(f"P_0 (cold start)[vtt] = {P_0[vtt_idx]:.15e}")
print(f"P_0 (warm start)[vtt] = {P_0_warm[vtt_idx]:.15e}")
print(f"|P_0_cold - P_0_warm| = {np.max(np.abs(P_0 - P_0_warm)):.3e}")
print(f"|R_cold - R_warm|     = {np.max(np.abs(R_converged - R_warm)):.3e}")
if np.max(np.abs(P_0 - P_0_warm)) > 1e-14:
    print("*** WARM START CHANGES P_0 — this is the source of the FD paradox ***")
    # Use warm-start P_0 as baseline for FD
    P_0 = P_0_warm
    R_converged = R_warm
    print("Switching to warm-start P_0 as baseline.")

eps_fd = 1e-7

R_0_full = proj._last_reaction_full.copy()

# Also call tangent_cone_split at this point (same data as FD baseline)
proj._last_reaction_full[:] = R_converged
_ = proj.project(y0.copy(), cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                  prev_state=y0.copy(), step_size=h)
Dproj2, Dstate2 = proj.tangent_cone_split(
    cand_0.copy(), y0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
    prev_state=y0.copy(), step_size=h,
)
Dstate2_d = Dstate2.toarray() if sp.issparse(Dstate2) else np.asarray(Dstate2)
print(f"\n=== tangent_cone_split at FD baseline ===")
print(f"  Dstate[vtt, slip] = {Dstate2_d[vtt_idx, slip_idx]:.6e}")
print(f"  Dstate[slip, slip] = {Dstate2_d[slip_idx, slip_idx]:.6e}")

# Get data from _last_solve_data for dR/dmu debugging
tc_data = proj._last_solve_data
tc_R0 = tc_data["solved"]["reaction"].copy()
tc_G = tc_data["solved"]["G_state"].copy()
print(f"  R_0 from _last_solve_data: {tc_R0}")
print(f"  R_0 from FD baseline:      {R_0_full}")

for j, label in [(vtt_idx, "vtt"), (slip_idx, "slip")]:
    eps_j = eps_fd * max(1.0, abs(y0[j]))
    y_p = y0.copy()
    y_p[j] += eps_j
    F_p = implicit_eq(y_p)
    cand_p = y_p - 1.0 * F_p

    # Case A: perturb both (standard FD)
    proj._last_reaction_full[:] = R_converged
    P_both = proj.project(y_p, cand_p, rhok=1.0, t=t_new, Fk_val=F_p,
                          prev_state=y0.copy(), step_size=h)
    R_both = proj._last_reaction_full.copy()

    # Case B: perturb only candidate, keep current_state = y0
    proj._last_reaction_full[:] = R_converged
    P_cand_only = proj.project(y0.copy(), cand_p, rhok=1.0, t=t_new, Fk_val=F_p,
                               prev_state=y0.copy(), step_size=h)

    # Case C: perturb only current_state, keep candidate = cand_0
    proj._last_reaction_full[:] = R_converged
    P_state_only = proj.project(y_p, cand_0.copy(), rhok=1.0, t=t_new, Fk_val=F_y,
                                prev_state=y0.copy(), step_size=h)
    R_state = proj._last_reaction_full.copy()

    dP_both = (P_both - P_0) / eps_j
    dP_cand = (P_cand_only - P_0) / eps_j
    dP_state = (P_state_only - P_0) / eps_j

    print(f"\n  Perturbing DOF {j} ({label}):")
    print(f"    P_0[vtt]         = {P_0[vtt_idx]:.14e}")
    print(f"    P_both[vtt]      = {P_both[vtt_idx]:.14e}")
    print(f"    P_cand_only[vtt] = {P_cand_only[vtt_idx]:.14e}")
    print(f"    P_state_only[vtt]= {P_state_only[vtt_idx]:.14e}")
    for r, rlabel in [(vtt_idx, "vtt"), (slip_idx, "slip")]:
        print(f"    dP[{rlabel}]/d{label}: both={dP_both[r]:.6e}, "
              f"cand_only={dP_cand[r]:.6e}, state_only={dP_state[r]:.6e}")
    if label == "slip":
        dR = (R_state - R_0_full) / eps_j
        print(f"    dR/dslip (FD): {dR}")
        dP_via_G = tc_G @ dR
        print(f"    G @ dR/dslip [vtt] = {dP_via_G[vtt_idx]:.6e} (should match dP_state[vtt]={dP_state[vtt_idx]:.6e})")
        print(f"    Dstate[vtt, slip]  = {Dstate2_d[vtt_idx, slip_idx]:.6e} (from tangent_cone_split)")

        # Direction-dependent dR/dmu test
        print(f"\n    === dR/dmu direction test ===")
        tc_solved = tc_data["solved"]
        R_base = tc_solved["reaction"].copy()
        mu_base = np.asarray(tc_solved["mu"], dtype=float)
        alpha_base = np.asarray(tc_solved["alpha"], dtype=float)
        eps_mu = 1e-7
        for sign, slabel in [(+1, "+eps"), (-1, "-eps")]:
            mu_test = mu_base.copy()
            alpha_test = alpha_base.copy()
            mu_test[0] += sign * eps_mu
            alpha_test[0] = mu_test[0]  # beta=0
            test_model = {
                "u_free": tc_solved["u_free"],
                "U_y": tc_solved["U_y"],
                "G_state": tc_G,
                "mu": mu_test,
                "alpha": alpha_test,
                "block_slices": tc_solved["block_slices"],
                "warm_start": R_base.copy(),
                "offset": tc_solved["offset"],
            }
            solved_test = proj._solve_local_problem(test_model)
            dR_mu = (solved_test["reaction"] - R_base) / (sign * eps_mu)
            GdR = tc_G @ dR_mu
            print(f"    mu{slabel}: dR/dmu = {dR_mu}, G@dR/dmu[vtt] = {GdR[vtt_idx]:.6e}, iters={solved_test['info']['iterations']}, res={solved_test['info']['residual']:.3e}")

        # Analytical dR/dmu comparison
        print(f"\n    === Analytical dR/dmu ===")
        J_nat = tc_solved["J_nat"]
        J_proj_c = tc_solved["J_proj"]
        u_conv = tc_solved["u"]
        u_hat_conv = tc_solved["u_hat"]
        rho_full = tc_solved["rho_full"]
        offset_s = tc_solved["offset"]
        block_slices_s = tc_solved["block_slices"]
        sl0 = block_slices_s[0]

        R_eff = R_base + offset_s if offset_s is not None else R_base
        z_conv = R_eff - rho_full * u_hat_conv

        print(f"    R_base = {R_base}")
        print(f"    R_eff  = {R_eff}")
        print(f"    u_hat  = {u_hat_conv}")
        print(f"    z_conv = {z_conv}")
        print(f"    rho    = {rho_full}")

        z_blk = z_conv[sl0]
        s_z, w_z = float(z_blk[0]), z_blk[1:]
        r_z = float(np.linalg.norm(w_z))
        mu_k = float(mu_base[0])
        lp = s_z + mu_k * r_z
        lm = s_z - r_z / mu_k
        print(f"    z_blk = {z_blk}, s={s_z:.6e}, r={r_z:.6e}")
        print(f"    lam_+ = {lp:.6e}, lam_- = {lm:.6e} (region={'interior' if lm>=0 else 'boundary' if lp>0 else 'polar'})")

        dproj_dmu_blk = proj._dproj_dmu_block(z_blk, mu_k)
        print(f"    dProj/dmu (analytical) = {dproj_dmu_blk}")

        # FD check of dProj/dmu
        from solve_nivp.projections import MuScaledSOCProjection
        p0 = MuScaledSOCProjection._proj_mu_scaled_soc(z_blk, mu_k)
        p_p = MuScaledSOCProjection._proj_mu_scaled_soc(z_blk, mu_k + 1e-7)
        p_m = MuScaledSOCProjection._proj_mu_scaled_soc(z_blk, mu_k - 1e-7)
        dp_fwd = (p_p - p0) / 1e-7
        dp_bwd = (p0 - p_m) / 1e-7
        dp_cen = (p_p - p_m) / 2e-7
        print(f"    dProj/dmu (FD fwd) = {dp_fwd}")
        print(f"    dProj/dmu (FD bwd) = {dp_bwd}")
        print(f"    dProj/dmu (FD cen) = {dp_cen}")

        u_t = u_conv[sl0.start + 1:sl0.stop]
        du_hat_dmu = np.zeros(R_base.size, dtype=float)
        du_hat_dmu[sl0.start] = float(np.linalg.norm(u_t))
        print(f"    ||u_t|| = {float(np.linalg.norm(u_t)):.6e}")

        dproj_dmu_full = np.zeros(R_base.size, dtype=float)
        dproj_dmu_full[sl0] = dproj_dmu_blk
        dF_dmu = -(J_proj_c @ (-rho_full * du_hat_dmu) + dproj_dmu_full)
        dR_dmu_ana = np.linalg.solve(J_nat, -dF_dmu)
        GdR_ana = tc_G @ dR_dmu_ana
        print(f"    dR/dmu (analytical) = {dR_dmu_ana}")
        print(f"    G@dR/dmu[vtt] (ana) = {GdR_ana[vtt_idx]:.6e}")
        print(f"    J_nat = {J_nat}")
        print(f"    cond(J_nat) = {np.linalg.cond(J_nat):.3e}")
        print(f"    dF/dmu = {dF_dmu}")

        # Verify IFT: F(R0 + eps*dR_ana, mu+eps) should ≈ 0
        from solve_nivp.desaxce_contact import _DeSaxceConeProjection
        for dr_label, dr_vec in [("ana", dR_dmu_ana), ("FD-", np.array([-1764.70574388, -2941.17671729]))]:
            R_test = R_base + 1e-7 * dr_vec
            u_test = tc_solved["u_free"] + tc_solved["W"] @ R_test
            u_hat_test, _ = proj._uhat_and_jac(u_test, alpha_base - 1e-7, block_slices_s)
            R_eff_test = R_test + offset_s if offset_s is not None else R_test
            z_test = R_eff_test - rho_full * u_hat_test
            p_test = proj._project_blocks(z_test, mu_base - 1e-7, block_slices_s)
            F_test = R_eff_test - p_test
            print(f"    F(R0+eps*dR_{dr_label}, mu-eps) = {F_test}, |F| = {np.linalg.norm(F_test):.3e}")

# Run the full solve with 1 step
print(f"\n{'='*60}")
print("Full solve_nivp: 1 step with SSN")
print(f"{'='*60}")

import solve_nivp

n_total = cs_red.y0.size
n_phys_proj = proj.n_phys
nl_atol = np.full(n_total, 1e-8)
nl_rtol = np.full(n_total, 1e-6)

t_arr, y_arr, h_arr, fk_arr, info_arr = solve_nivp.solve_nivp(
    fun=cs_red.rhs,
    t_span=(0.0, h),
    y0=cs_red.y0,
    method="backward_euler",
    projection=cs_red.projection,
    solver="semismooth_newton",
    solver_opts={
        "tol": 1e-8,
        "max_iter": 50,
        "globalization": "none",
        "linear_solver": "splu",
        "rhs_jac": cs_red.rhs_jac,
        "adaptive_lam": False,
    },
    adaptive=False,
    h0=h,
    integrator_opts=cs_red.integrator_opts,
    component_slices=cs_red.component_slices,
    A=cs_red.A,
    store_fk=False,
    nl_atol=nl_atol,
    nl_rtol=nl_rtol,
)

print(f"Steps taken: {len(t_arr) - 1}")
for i, (err, success, iters) in enumerate(info_arr):
    print(f"  step {i}: success={success}, iters={iters}, err={err:.3e}")
