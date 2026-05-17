#!/usr/bin/env python
"""Audit the SOC natural-map residual on a one-step sliding-block case.

This script is aimed at the specific debugging question:

* for ``mu = 0``, does the SOC backend's reaction block reduce to the same
  frictionless contact residual as the Alart-Curnier formulation?
* if the final iterate is physically plausible but the solver reports
  non-convergence, which block of the residual is actually large?

It reconstructs, on the accepted one-step iterate:

* the true backward-Euler residual ``F(y)``
* the full natural-map residual ``Phi(y) = y - Pi(y - lam F(y))``
* a frictionless Alart-Curnier-style residual on the reaction block

and reports norms blockwise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PORO_ROOT = Path("/home/david/Documents/Poroelasticity")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(REPO_ROOT / "src"))
if PORO_ROOT.exists():
    sys.path.insert(0, str(PORO_ROOT))

import solve_nivp
from sliding_block_one_step_patch_test import build_demo_contact_system


@dataclass
class VectorNorms:
    inf_norm: float
    rms_norm: float


@dataclass
class BlockAudit:
    diff_phys: VectorNorms
    alg_phys: VectorNorms
    react: VectorNorms


@dataclass
class AuditResult:
    success: bool
    iterations: int
    solver_error: float | None
    step_size: float
    t_final: float
    active_contacts: int
    open_contacts: int
    max_gap: float
    min_gap: float
    true_be_full: VectorNorms
    true_be_blocks: BlockAudit
    natural_map_full: VectorNorms
    natural_map_blocks: BlockAudit
    ac_mu0_react: VectorNorms
    active_contact_equivalence_inf: float
    active_contact_equivalence_rms: float


def _norms(vec: np.ndarray) -> VectorNorms:
    arr = np.asarray(vec, dtype=float).ravel()
    if arr.size == 0:
        return VectorNorms(0.0, 0.0)
    return VectorNorms(
        inf_norm=float(np.max(np.abs(arr))),
        rms_norm=float(np.sqrt(np.mean(arr * arr))),
    )


def _physical_q_indices(projection, n_phys: int) -> np.ndarray:
    q_mask = np.zeros(n_phys, dtype=bool)
    alg = getattr(projection, "_alg", None)
    if alg is not None:
        for blk in getattr(alg, "_blocks", []):
            q_mask[blk.q_slice] = True
    return np.flatnonzero(q_mask)


def _step_residual(y: np.ndarray, y_prev: np.ndarray, t: float, h: float, ctx: dict) -> np.ndarray:
    res = (ctx["cs"].A @ ((y - y_prev) / h)) - ctx["cs"].rhs(t, y, y_prev, h)
    return np.asarray(res, dtype=float).ravel()


def _natural_map_residual(
    y: np.ndarray,
    F_y: np.ndarray,
    y_prev: np.ndarray,
    t: float,
    h: float,
    ctx: dict,
    lam: float,
) -> np.ndarray:
    proj = ctx["cs"].projection
    cand = y - lam * F_y
    proj_val = proj.project(
        y,
        cand,
        rhok=lam,
        t=t,
        Fk_val=F_y,
        prev_state=y_prev,
        step_size=h,
    )
    return np.asarray(y - proj_val, dtype=float).ravel()


def _frictionless_ac_reaction_residual(
    y: np.ndarray,
    F_y: np.ndarray,
    ctx: dict,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_phys = ctx["n_orig"]
    n_c = ctx["n_c"]
    r = np.asarray(y[n_phys:], dtype=float).ravel()
    uhat = np.asarray(F_y[n_phys:], dtype=float).ravel()
    gaps = np.asarray(ctx["gap_func"](y[:n_phys], 0.0), dtype=float).ravel()
    gap_tol = float(getattr(getattr(ctx["cs"].projection, "_soc", ctx["cs"].projection), "gap_tol", 0.0))
    active = gaps <= gap_tol

    out = np.zeros_like(r)
    for k in range(n_c):
        sl = slice(2 * k, 2 * k + 2)
        r_blk = r[sl]
        u_blk = uhat[sl]
        if not active[k]:
            out[sl] = r_blk
            continue
        rn = float(r_blk[0])
        un = float(u_blk[0])
        # mu = 0 -> tangential ball has radius 0, so tangential residual is -r_t.
        out[sl] = np.array(
            [
                max(rn - lam * un, 0.0) - rn,
                -float(r_blk[1]),
            ],
            dtype=float,
        )
    return out, active


def run_audit(
    *,
    mu_friction: float,
    initial_gap_phys: float,
    t_end_hours: float,
    reverse_gap_sign: bool,
    top_v1_rate: float,
    top_v2_rate: float,
    rho_g: float,
    n_elem: int,
    element_type: str,
    lam: float,
) -> AuditResult:
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = build_demo_contact_system(
            mu_friction=mu_friction,
            initial_gap_phys=initial_gap_phys,
            reverse_gap_sign=reverse_gap_sign,
            top_v1_rate=top_v1_rate,
            top_v2_rate=top_v2_rate,
            rho_g=rho_g,
            n_elem=n_elem,
            element_type=element_type,
        )

    tmax = t_end_hours / ctx["poro"].get_scales()[0]
    solver_opts = dict(ctx["solver_opts_contact"])
    solver_opts.update(
        {
            "max_iter": 30,
            "adaptive_lam": False,
            "lam_update_strategy": "none",
            "globalization": "linesearch",
            "use_broyden": False,
            "linear_solver": "splu",
            "sparse": True,
            "lam": lam,
        }
    )

    out = solve_nivp.solve_nivp(
        fun=ctx["cs"].rhs,
        t_span=(0.0, tmax),
        y0=ctx["cs"].y0,
        method="backward_euler",
        projection=ctx["cs"].projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts=solver_opts,
        adaptive=False,
        h0=tmax,
        integrator_opts=ctx["cs"].integrator_opts,
        nl_atol=ctx["nl_atol_contact"],
        nl_rtol=1.0e-6,
        component_slices=ctx["cs"].component_slices,
        verbose=False,
        A=ctx["cs"].A,
        store_fk=False,
    )

    t_vals, y_vals, _, _, error_estimates = out
    solver_error, success, iterations = error_estimates[-1]
    y_prev = np.asarray(y_vals[-2], dtype=float)
    y_fin = np.asarray(y_vals[-1], dtype=float)
    t_fin = float(t_vals[-1])
    h_step = float(t_vals[-1] - t_vals[-2])

    F_be = _step_residual(y_fin, y_prev, t_fin, h_step, ctx)
    Phi = _natural_map_residual(y_fin, F_be, y_prev, t_fin, h_step, ctx, lam=lam)
    ac_react, active = _frictionless_ac_reaction_residual(y_fin, F_be, ctx, lam=lam)

    n_phys = ctx["n_orig"]
    react_slice = slice(n_phys, y_fin.size)
    q_idx = _physical_q_indices(ctx["cs"].projection, n_phys)
    q_mask = np.zeros(n_phys, dtype=bool)
    q_mask[q_idx] = True
    diff_idx = np.flatnonzero(~q_mask)

    be_diff = F_be[diff_idx]
    be_alg = F_be[q_idx]
    be_react = F_be[react_slice]

    phi_diff = Phi[diff_idx]
    phi_alg = Phi[q_idx]
    phi_react = Phi[react_slice]

    # On active contacts with mu=0, the AC frictionless reaction residual should
    # match the SOC natural-map reaction residual up to sign.
    active_contact_mask = np.repeat(active, 2)
    equiv = phi_react[active_contact_mask] + ac_react[active_contact_mask]

    gaps = np.asarray(ctx["gap_func"](y_fin[:n_phys], t_fin), dtype=float).ravel()
    gap_tol = float(getattr(getattr(ctx["cs"].projection, "_soc", ctx["cs"].projection), "gap_tol", 0.0))
    open_mask = gaps > gap_tol

    return AuditResult(
        success=bool(success),
        iterations=int(iterations),
        solver_error=float(solver_error) if solver_error is not None else None,
        step_size=h_step,
        t_final=t_fin,
        active_contacts=int(np.count_nonzero(active)),
        open_contacts=int(np.count_nonzero(open_mask)),
        max_gap=float(np.max(gaps)) if gaps.size else 0.0,
        min_gap=float(np.min(gaps)) if gaps.size else 0.0,
        true_be_full=_norms(F_be),
        true_be_blocks=BlockAudit(
            diff_phys=_norms(be_diff),
            alg_phys=_norms(be_alg),
            react=_norms(be_react),
        ),
        natural_map_full=_norms(Phi),
        natural_map_blocks=BlockAudit(
            diff_phys=_norms(phi_diff),
            alg_phys=_norms(phi_alg),
            react=_norms(phi_react),
        ),
        ac_mu0_react=_norms(ac_react),
        active_contact_equivalence_inf=float(np.max(np.abs(equiv))) if equiv.size else 0.0,
        active_contact_equivalence_rms=float(np.sqrt(np.mean(equiv * equiv))) if equiv.size else 0.0,
    )


def _print_norms(name: str, norms: VectorNorms) -> None:
    print(f"  {name:<18} inf={norms.inf_norm:.4e}   rms={norms.rms_norm:.4e}")


def _print_block_audit(title: str, blk: BlockAudit) -> None:
    print(title)
    _print_norms("diff phys", blk.diff_phys)
    _print_norms("alg phys", blk.alg_phys)
    _print_norms("react", blk.react)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--initial-gap", type=float, default=0.0)
    parser.add_argument("--t-end-hours", type=float, default=0.01)
    parser.add_argument("--top-v1-rate", type=float, default=0.0)
    parser.add_argument("--top-v2-rate", type=float, default=-5.0e-6)
    parser.add_argument("--rho-g", type=float, default=0.0)
    parser.add_argument("--n-elem", type=int, default=12)
    parser.add_argument("--element-type", choices=("tri", "quad"), default="tri")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--reverse-gap-sign", action="store_true")
    args = parser.parse_args()

    res = run_audit(
        mu_friction=args.mu,
        initial_gap_phys=args.initial_gap,
        t_end_hours=args.t_end_hours,
        reverse_gap_sign=bool(args.reverse_gap_sign),
        top_v1_rate=args.top_v1_rate,
        top_v2_rate=args.top_v2_rate,
        rho_g=args.rho_g,
        n_elem=args.n_elem,
        element_type=args.element_type,
        lam=args.lam,
    )

    print("SOC mu=0 natural-map audit")
    print(f"  mu = {args.mu}")
    print(f"  success = {res.success}")
    print(f"  iterations = {res.iterations}")
    print(f"  solver_error = {res.solver_error}")
    print(f"  t_final = {res.t_final:.6e}")
    print(f"  h_step = {res.step_size:.6e}")
    print(f"  gap range = [{res.min_gap:+.4e}, {res.max_gap:+.4e}]")
    print(f"  active contacts = {res.active_contacts}")
    print(f"  open contacts = {res.open_contacts}")

    print("\nTrue backward-Euler residual")
    _print_norms("full", res.true_be_full)
    _print_block_audit("by block", res.true_be_blocks)

    print("\nNatural-map residual")
    _print_norms("full", res.natural_map_full)
    _print_block_audit("by block", res.natural_map_blocks)

    print("\nFrictionless AC-style reaction residual")
    _print_norms("react only", res.ac_mu0_react)
    print(
        "  active-block equivalence "
        f"||Phi_react + f_AC||_inf = {res.active_contact_equivalence_inf:.4e}"
    )
    print(
        "  active-block equivalence "
        f"||Phi_react + f_AC||_rms = {res.active_contact_equivalence_rms:.4e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
