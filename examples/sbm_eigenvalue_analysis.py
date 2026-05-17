"""
Standalone eigenvalue diagnostic for the SBM contact dynamics.

Two complementary methods, both run by default (``--method both``):

  1. ``direct``: assemble the descriptor mass A_dyn and the linearised
     RHS Jacobian J at a chosen snapshot of the pilot trajectory, then
     solve the generalised eigenproblem
            J · v = λ · A_dyn · v
     via ARPACK shift-invert at a user-chosen σ (default 0.05).
     This is the analytical eigenvalue of the discrete operator —
     not affected by trajectory truncation, IC selection, or DMD-rank
     choice.  Use this for definitive answers.

  2. ``dmd``: snapshot-based Dynamic Mode Decomposition on the post-ramp
     part of the velocity trajectory.  Empirical, observes only modes
     excited by the IC.  Useful as a cross-check.

Both report the top-N eigenvalues sorted by descending Re(λ).  A positive
Re(λ) signals an exponentially-growing mode (linear instability of the
discrete dynamic operator).

Usage
-----
    # both methods, default settings
    python examples/sbm_eigenvalue_analysis.py \
        --N 40 --use-sbm True --taylor True --sbm True

    # only the direct ARPACK method (faster — skips the DMD post-process)
    python examples/sbm_eigenvalue_analysis.py \
        --N 40 --taylor True --sbm True --method direct

    # cross-check: SBM stress-correction term off, expect ✓ STABLE
    python examples/sbm_eigenvalue_analysis.py \
        --N 40 --taylor True --sbm False

The script exec()s the embedded-crack notebook (converted to a script via
``jupyter nbconvert --to script``) with the chosen flags substituted in,
captures the velocity trajectory, and applies the chosen method(s).

Tunable: --shift (ARPACK σ), --linearise-time (snapshot for direct),
--n-eigs (number of eigenvalues), --rank (DMD truncation rank),
--tmax (pilot trajectory length).

Requires the conda env ``fem-env`` (PETSc, scikit-fem, poroelasticity).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time
import sys

import numpy as np


def _bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in ("true", "1", "yes", "y"):
        return True
    if s.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {s!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=30,
                   help="N_ELEM mesh refinement (default: 30)")
    p.add_argument("--use-sbm", type=_bool, default=True,
                   help="USE_SBM master toggle (False = conforming mesh)")
    p.add_argument("--taylor", type=_bool, default=False,
                   help="INCLUDE_TAYLOR (trial-side Taylor shift)")
    p.add_argument("--taylor-test", type=_bool, default=False,
                   help="INCLUDE_TAYLOR_TEST (no-op under lumped_coupling=True)")
    p.add_argument("--sbm", type=_bool, default=False,
                   help="INCLUDE_SBM (n_perp stress correction term)")
    p.add_argument("--taylor-method", choices=("nodal", "l2_project"),
                   default="nodal",
                   help="taylor_method for the assembler")
    p.add_argument("--tmax", type=float, default=10.0,
                   help="TMAX_PHYS in seconds for the pilot trajectory")
    p.add_argument("--t-linear-start", type=float, default=8.0,
                   help="dimensionless time after which dynamics are linear")
    p.add_argument("--rank", type=int, default=8,
                   help="DMD truncation rank")
    p.add_argument("--method", choices=("dmd", "direct", "both"),
                   default="both",
                   help="eigenvalue method: DMD on trajectory, direct ARPACK "
                        "on the linearised Jacobian, or both")
    p.add_argument("--shift", type=float, default=0.05,
                   help="shift-invert σ for direct ARPACK (near expected "
                        "unstable mode; default 0.05)")
    p.add_argument("--n-eigs", type=int, default=10,
                   help="number of eigenvalues to extract via ARPACK")
    p.add_argument("--linearise-time", type=float, default=20.0,
                   help="dimensionless time at which to linearise for the "
                        "direct method (snapshot from the pilot trajectory)")
    p.add_argument("--label", type=str, default="",
                   help="optional label suffix for saved trajectory")
    p.add_argument("--script-path", type=str,
                   default=os.path.expanduser(
                       "~/Documents/Solve_ivp_ns/examples/"
                       "embedded_crack_mohr_coulomb_ncp.ipynb"),
                   help="path to embedded crack notebook (.ipynb)")
    p.add_argument("--out-dir", type=str, default="/tmp",
                   help="directory for saved npz files")
    return p.parse_args()


def _convert_notebook_to_script(ipynb_path: str) -> str:
    """Convert a Jupyter notebook to a single Python script string."""
    import nbformat
    from nbconvert import PythonExporter
    nb = nbformat.read(ipynb_path, as_version=4)
    exporter = PythonExporter()
    src, _ = exporter.from_notebook_node(nb)
    return src


def _patch_source(src: str, args: argparse.Namespace) -> str:
    """Apply the parameter substitutions to the notebook script."""
    def _sub(text, old, new):
        if text.count(old) != 1:
            raise RuntimeError(f"expected exactly one match for {old!r}")
        return text.replace(old, new, 1)

    src = _sub(src, "N_ELEM     = 30",                f"N_ELEM     = {args.N}")
    src = _sub(src, "USE_SBM               = True",   f"USE_SBM               = {args.use_sbm}")
    src = _sub(src, "INCLUDE_SBM           = False",  f"INCLUDE_SBM           = {args.sbm}")
    src = _sub(src, "INCLUDE_TAYLOR        = False",  f"INCLUDE_TAYLOR        = {args.taylor}")
    src = _sub(src, "INCLUDE_TAYLOR_TEST   = False",  f"INCLUDE_TAYLOR_TEST   = {args.taylor_test}")
    src = _sub(src, "TMAX_PHYS = 30.0",               f"TMAX_PHYS = {args.tmax}")
    inject = f"taylor_method='{args.taylor_method}',"
    src = src.replace("include_sbm=INCLUDE_SBM,",
                      f"include_sbm=INCLUDE_SBM, {inject}")
    src = src.replace("include_sbm=False,",
                      f"include_sbm=False, {inject}")
    return src


def _run_pilot(args: argparse.Namespace, return_solver: bool = False) -> dict:
    """Execute the patched notebook script up to the sliding solve.
    Returns a dict with t, v_hist, etc.  If return_solver=True, also
    returns the assembled CGPoroelastostatics solver and ContactSystem
    for use in the direct eigenvalue analysis."""
    src = _convert_notebook_to_script(args.script_path)
    src = _patch_source(src, args)

    # Cut just after the sliding solve metrics print, before phase-2 hold cells.
    end_marker = "sliding_gap = np.asarray(y_sliding[:, jmpu_n_idx_s]"
    end_idx = src.find(end_marker)
    if end_idx < 0:
        raise RuntimeError("could not locate sliding-solve end marker in notebook")
    src_short = src[:end_idx]

    import matplotlib
    matplotlib.use("Agg")

    g = {"__name__": "__main__"}
    print(f"[pilot] running with N={args.N}, use_sbm={args.use_sbm}, "
          f"T={args.taylor}, Tt={args.taylor_test}, S={args.sbm}, "
          f"taylor_method={args.taylor_method}, TMAX_PHYS={args.tmax}")
    t0 = time.time()
    exec(compile(src_short, "<embedded_crack>", "exec"), g)
    elapsed = time.time() - t0
    print(f"[pilot] solve completed in {elapsed:.1f}s")

    t_arr = np.asarray(g["t_sliding"], dtype=float)
    y_arr = np.asarray(g["y_sliding"], dtype=float)
    n_phys = int(g["n_phys_s"])
    n_base = int(g["n_base_s"])
    v_hist = y_arr[:, n_base:n_phys]
    out = dict(
        t=t_arr, y_hist=y_arr, v_hist=v_hist,
        u_inf=np.max(np.abs(y_arr[:, :n_base]), axis=1),
        v_inf=np.max(np.abs(v_hist), axis=1),
        Nu=v_hist.shape[1],
        n_phys=n_phys, n_base=n_base,
    )
    if return_solver:
        out["poro_s"] = g["poro_s"]
        out["cs_s"] = g["cs_s"]
        out["A_dyn_s"] = g["A_dyn_s"]
    return out


def dmd(t: np.ndarray, V: np.ndarray, t_start: float, rank: int) -> dict:
    """Apply rank-r DMD to V (Nu × n_t) on the slice t >= t_start.

    Resamples to a uniform time grid via linear interpolation, then fits
    a rank-r linear operator A: V_{k+1} ≈ A V_k.  Returns continuous-time
    eigenvalues λ_j = log(μ_j)/Δt where μ_j are eigenvalues of the reduced
    operator.
    """
    mask = t >= t_start
    t_lin = t[mask]
    V_lin = V[mask].T   # shape (Nu, K)
    K = V_lin.shape[1]
    if K < rank + 5:
        raise RuntimeError(
            f"only {K} snapshots in linear regime; "
            f"need at least {rank + 5} for rank-{rank} DMD")

    n_uniform = K
    t_u = np.linspace(t_lin[0], t_lin[-1], n_uniform)
    dt = (t_u[-1] - t_u[0]) / (n_uniform - 1)
    V_u = np.empty((V_lin.shape[0], n_uniform))
    for i in range(V_lin.shape[0]):
        V_u[i] = np.interp(t_u, t_lin, V_lin[i])

    V_minus = V_u[:, :-1]
    V_plus = V_u[:, 1:]

    U_svd, s_svd, Vt_svd = np.linalg.svd(V_minus, full_matrices=False)
    r = min(rank, len(s_svd))
    U_r = U_svd[:, :r]; s_r = s_svd[:r]; W_r = Vt_svd[:r].T
    A_tilde = U_r.T @ V_plus @ W_r @ np.diag(1.0 / s_r)
    mu, _ = np.linalg.eig(A_tilde)
    lam = np.log(mu + 0j) / dt
    order = np.argsort(-lam.real)
    return dict(lam=lam[order], mu=mu[order],
                rank=r, dt=dt, t_start=t_start, n_snapshots=K,
                singular_values=s_svd[:r])


def direct_eigvals(traj: dict, args: argparse.Namespace) -> dict:
    """Direct ARPACK eigenvalue analysis of the linearised dynamic system.

    Builds the descriptor mass matrix and the linearised RHS Jacobian at a
    chosen snapshot from the pilot trajectory, then solves the generalised
    eigenproblem ``J · v = λ · A_dyn · v`` using shift-invert ARPACK with
    σ = ``args.shift``.

    Returns a dict with the eigenvalues sorted by descending Re(λ).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if "cs_s" not in traj:
        raise RuntimeError("direct method requires return_solver=True from pilot")

    t_arr  = traj["t"]
    y_hist = traj["y_hist"]
    cs_s   = traj["cs_s"]
    A_dyn  = traj["A_dyn_s"].tocsr()

    # Snapshot at the requested time (post-ramp, in sliding regime).
    i_lin = int(np.argmin(np.abs(t_arr - args.linearise_time)))
    t_lin = float(t_arr[i_lin])
    y_lin = y_hist[i_lin].copy()
    print(f"[direct] linearising at t = {t_lin:.3f} (snapshot {i_lin}/{len(t_arr)})")

    # Linearised RHS Jacobian via cs_s (the augmented contact system).
    J = cs_s.rhs_jac(t_lin, y_lin)
    if hasattr(J, "tocsr"):
        J = J.tocsr()

    # Descriptor matrix on the augmented system: A_dyn for physical DOFs,
    # zeros for the appended multiplier rows/cols.
    n_aug = cs_s.A.shape[0]
    if J.shape[0] != A_dyn.shape[0]:
        n_extra = n_aug - A_dyn.shape[0]
        A_pad = sp.bmat([
            [A_dyn,                                   sp.csr_matrix((A_dyn.shape[0], n_extra))],
            [sp.csr_matrix((n_extra, A_dyn.shape[0])), sp.csr_matrix((n_extra, n_extra))],
        ], format="csr")
    else:
        A_pad = A_dyn

    print(f"[direct] J shape={J.shape} (nnz={J.nnz}), "
          f"A_pad shape={A_pad.shape} (nnz={A_pad.nnz})")
    print(f"[direct] solving (J - σ A) z = A x with σ = {args.shift:+.4e} …")

    sigma = float(args.shift)
    M_op = (J - sigma * A_pad).tocsc()
    try:
        lu = spla.splu(M_op)
    except RuntimeError as exc:
        raise RuntimeError(
            f"SuperLU could not factor (J - σ·A_dyn) at σ={sigma}; "
            f"try a different --shift value: {exc}")

    OP = spla.LinearOperator(M_op.shape, matvec=lambda x: lu.solve(A_pad @ x))
    import time
    t0 = time.time()
    mu, V = spla.eigs(OP, k=args.n_eigs, which="LM",
                      maxiter=400, tol=1e-8)
    elapsed = time.time() - t0

    # ARPACK returns eigenvalues μ of shift-invert OP; recover λ via
    # μ = 1 / (λ - σ)  ⟹  λ = σ + 1/μ.
    lam = sigma + 1.0 / mu
    order = np.argsort(-lam.real)
    lam = lam[order]; V = V[:, order]
    print(f"[direct] ARPACK converged in {elapsed:.1f}s")
    return dict(lam=lam, eigvecs=V, t_lin=t_lin, sigma=sigma)


def report(args: argparse.Namespace, traj: dict, dmd_out: dict, direct_out=None) -> None:
    """Print a clean summary of the analysis."""
    t = traj["t"]; u_inf = traj["u_inf"]; v_inf = traj["v_inf"]
    print()
    print("=" * 72)
    print(f"  SBM eigenvalue analysis — N={args.N}, use_sbm={args.use_sbm}, "
          f"T={args.taylor}, S={args.sbm}, taylor_method={args.taylor_method}")
    print("=" * 72)
    print(f"  Pilot trajectory: {len(t)} snapshots, t_final={t[-1]:.2f} (nondim)")
    print(f"  Final norms: u_inf={u_inf[-1]:.3e}, v_inf={v_inf[-1]:.3e}")

    def _print_table(name: str, lam: np.ndarray) -> None:
        print(f"\n  {name} eigenvalues sorted by descending Re(λ):")
        print(f"  {'#':>3} {'Re(λ)':>14} {'Im(λ)':>14} {'doubling/decay':>20}")
        for j, l in enumerate(lam):
            re, im = float(l.real), float(l.imag)
            if re > 1e-9:
                tag = f"+{np.log(2)/re:>9.2f}  GROW"
            elif re < -1e-9:
                tag = f"-{np.log(2)/abs(re):>9.2f}  DECAY"
            else:
                tag = "marginal"
            print(f"  {j:3d} {re:>+14.6e} {im:>+14.6e} {tag:>20}")

    if dmd_out is not None:
        lam = dmd_out["lam"]
        print(f"\n  DMD on snapshots t >= {dmd_out['t_start']:.1f}  "
              f"(rank={dmd_out['rank']}, dt={dmd_out['dt']:.4f}, "
              f"n={dmd_out['n_snapshots']})")
        _print_table("DMD", lam)

    if direct_out is not None:
        lam_d = direct_out["lam"]
        print(f"\n  DIRECT (ARPACK shift-invert) at σ = {direct_out['sigma']:+.4e}, "
              f"linearised at t = {direct_out['t_lin']:.2f}")
        _print_table("DIRECT", lam_d)

    print()
    leading_dmd = dmd_out["lam"][0] if dmd_out is not None else None
    leading_direct = direct_out["lam"][0] if direct_out is not None else None
    leading = leading_direct if leading_direct is not None else leading_dmd
    if leading is None:
        return
    re_lead = float(leading.real)
    if re_lead > 1e-9:
        print(f"  ✗ UNSTABLE: leading mode has Re(λ) = {re_lead:+.5e}, "
              f"doubling time {np.log(2)/re_lead:.2f} nondim units.")
    elif re_lead < -1e-9:
        print(f"  ✓ STABLE: every mode has Re(λ) < 0; "
              f"slowest decay = {np.log(2)/abs(re_lead):.2f} nondim units.")
    else:
        print(f"  ~ MARGINAL: leading mode has Re(λ) ≈ 0 ({re_lead:+.2e}).")

    if dmd_out is not None and direct_out is not None:
        re_dmd = float(dmd_out["lam"][0].real)
        re_dir = float(direct_out["lam"][0].real)
        diff = abs(re_dmd - re_dir)
        if diff < 0.01 * max(abs(re_dmd), abs(re_dir), 1e-10):
            print(f"  Cross-check: DMD ({re_dmd:+.4e}) and DIRECT ({re_dir:+.4e}) agree.")
        else:
            print(f"  WARNING: DMD ({re_dmd:+.4e}) and DIRECT ({re_dir:+.4e}) disagree.")
            print(f"    Possible causes: linearisation point not in linear regime, "
                  f"DMD truncation rank too low, or shift σ missed the dominant mode.")
    print()


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    need_solver = args.method in ("direct", "both")
    traj = _run_pilot(args, return_solver=need_solver)

    dmd_out = None
    direct_out = None
    if args.method in ("dmd", "both"):
        dmd_out = dmd(traj["t"], traj["v_hist"],
                      t_start=args.t_linear_start, rank=args.rank)
    if args.method in ("direct", "both"):
        try:
            direct_out = direct_eigvals(traj, args)
        except Exception as exc:
            print(f"[direct] failed: {exc}")
            print(f"[direct] try a different --shift value (current: {args.shift})")

    label_part = f"_{args.label}" if args.label else ""
    out_path = out_dir / (
        f"eig_N{args.N}_USE{int(args.use_sbm)}"
        f"_T{int(args.taylor)}_Tt{int(args.taylor_test)}_S{int(args.sbm)}"
        f"_method-{args.taylor_method}{label_part}.npz")
    save_kwargs = dict(
        t=traj["t"], u_inf=traj["u_inf"], v_inf=traj["v_inf"],
        v_hist=traj["v_hist"], config=vars(args))
    if dmd_out is not None:
        save_kwargs.update(
            dmd_eigvals=dmd_out["lam"],
            dmd_singular_values=dmd_out["singular_values"],
            dmd_dt=dmd_out["dt"], dmd_t_start=dmd_out["t_start"])
    if direct_out is not None:
        save_kwargs.update(
            direct_eigvals=direct_out["lam"],
            direct_t_lin=direct_out["t_lin"],
            direct_sigma=direct_out["sigma"])
    np.savez(out_path, **save_kwargs)
    print(f"  Saved to {out_path}")

    report(args, traj, dmd_out, direct_out)
    leading = (direct_out["lam"][0] if direct_out is not None
               else dmd_out["lam"][0] if dmd_out is not None
               else 0.0)
    return 0 if float(np.real(leading)) <= 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
