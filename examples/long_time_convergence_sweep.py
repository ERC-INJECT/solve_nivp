"""
Long-time convergence sweep for the embedded-crack Mohr-Coulomb notebook.

Sweeps four axes holding ``include_sbm=False``, ``include_taylor_test=False``,
``USE_SBM=True`` fixed:

    N               in {30, 40, 60}
    lumped_coupling in {True, 'consistent'}
    taylor_method   in {'nodal', 'l2_project'}
    include_taylor  in {True, False}

For each (N, lumped, method, taylor) configuration the script:

    1. Patches the notebook with the chosen flags.
    2. Runs the full pilot trajectory through the L2 vs Pollard-Segall cell
       (the truncation is extended past the existing
       ``sbm_eigenvalue_analysis._run_pilot`` cut).
    3. Records the final ``u_inf``, the Pollard L2 error / reference / relative
       value, and the leading DMD eigenvalue on the post-ramp velocity history.
    4. Writes one row to ``out_dir/long_time_convergence_sweep.csv`` and saves
       the velocity history + eigenvalues to a per-config ``.npz``.

Results are appended row-by-row, so the sweep can be resumed after an
interruption: rerunning skips configurations whose CSV row is already
present unless ``--force`` is passed.

Usage
-----

    # full sweep (24 configs, ~3-4 h at TMAX=30)
    python examples/long_time_convergence_sweep.py --tmax 30

    # smoke test on a single config (1-2 min)
    python examples/long_time_convergence_sweep.py --tmax 10 \
        --N 30 --lumped True --method nodal --taylor True

    # resume after interruption (skips completed rows)
    python examples/long_time_convergence_sweep.py --tmax 30
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
import traceback
from typing import Any

import numpy as np

# Reuse the harness from sbm_eigenvalue_analysis without duplicating it.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import sbm_eigenvalue_analysis as sea  # noqa: E402


_NOTEBOOK = str(_HERE / "embedded_crack_mohr_coulomb_ncp.ipynb")
_LUMPED_VALUES = {
    "True": True,
    "False": False,
    "consistent": "consistent",
}


# --------------------------------------------------------------------------- #
# Patching                                                                    #
# --------------------------------------------------------------------------- #

_PARAM_RE = {
    "N_ELEM": r"^N_ELEM\s*=\s*[^\n#]+",
    "USE_SBM": r"^USE_SBM\s*=\s*[^\n#]+",
    "INCLUDE_SBM": r"^INCLUDE_SBM\s*=\s*[^\n#]+",
    "INCLUDE_TAYLOR": r"^INCLUDE_TAYLOR\s*=\s*[^\n#]+",
    "INCLUDE_TAYLOR_TEST": r"^INCLUDE_TAYLOR_TEST\s*=\s*[^\n#]+",
    "TMAX_PHYS": r"^TMAX_PHYS\s*=\s*[^\n#]+",
}


def _patch_source(src: str, *, N: int, tmax: float,
                  taylor: bool, taylor_method: str,
                  lumped_coupling: Any) -> str:
    """Apply parameter substitutions to the notebook script.

    Uses regex-based substitution so the patcher doesn't depend on the
    notebook's current default values.  Also injects ``taylor_method`` and
    ``lumped_coupling`` keywords into every ``CGPoroelastostatics(...)``
    call (identified by the ``include_sbm=`` keyword).
    """
    import re

    def _sub_param(text: str, name: str, value: str) -> str:
        pattern = _PARAM_RE[name]
        m = re.search(pattern, text, flags=re.MULTILINE)
        if m is None:
            raise RuntimeError(f"could not locate {name} assignment")
        return text[:m.start()] + f"{name} = {value}" + text[m.end():]

    src = _sub_param(src, "N_ELEM",              str(N))
    src = _sub_param(src, "USE_SBM",             "True")
    src = _sub_param(src, "INCLUDE_SBM",         "False")
    src = _sub_param(src, "INCLUDE_TAYLOR",      str(bool(taylor)))
    src = _sub_param(src, "INCLUDE_TAYLOR_TEST", "False")
    src = _sub_param(src, "TMAX_PHYS",           f"{tmax}")

    # Inject taylor_method and lumped_coupling into every CGPoroelastostatics
    # constructor call (identified by the include_sbm= keyword).
    inject = (f"taylor_method='{taylor_method}', "
              f"lumped_coupling={lumped_coupling!r},")
    src = src.replace("include_sbm=INCLUDE_SBM,",
                      f"include_sbm=INCLUDE_SBM, {inject}")
    src = src.replace("include_sbm=False,",
                      f"include_sbm=False, {inject}")
    return src


# --------------------------------------------------------------------------- #
# Pilot run with extended cut to capture L2 error                             #
# --------------------------------------------------------------------------- #

# The default cut in sbm_eigenvalue_analysis ends BEFORE the Pollard L2
# evaluation cell.  Cut at the start of the energy-balance cell that
# follows so that ``err_l2`` and ``ref_l2`` are computed but the heavier
# Jacobian-based energy diagnostics are skipped.
_L2_END_MARKER = "y_hist_s = np.asarray(y_sliding, dtype=float)"


def _run_pilot_with_l2(*, N: int, tmax: float, taylor: bool,
                       taylor_method: str, lumped_coupling: Any) -> dict:
    """Execute the patched notebook through the Pollard L2 cell.

    Returns
    -------
    dict
        ``{t, v_hist, u_inf, v_inf, Nu, n_phys, n_base,
           err_l2, ref_l2, rel_l2, slip_phys, slip_anal_phys, s_param}``
    """
    src = sea._convert_notebook_to_script(_NOTEBOOK)
    src = _patch_source(src, N=N, tmax=tmax, taylor=taylor,
                        taylor_method=taylor_method,
                        lumped_coupling=lumped_coupling)

    # Cut at the start of the energy-balance cell, which is right after the
    # L2-error print.  This keeps err_l2/ref_l2 in scope but drops the heavy
    # energy diagnostics that would slow the sweep.
    end_idx = src.find(_L2_END_MARKER)
    if end_idx < 0:
        raise RuntimeError("could not locate energy-balance marker in notebook")
    src_short = src[:end_idx]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None  # silence Agg show() warnings

    g: dict[str, Any] = {"__name__": "__main__"}
    print(f"  [pilot] N={N}, lumped={lumped_coupling!r}, "
          f"method={taylor_method}, T={taylor}, TMAX={tmax}")
    t0 = time.time()
    try:
        exec(compile(src_short, "<embedded_crack>", "exec"), g)
    except BaseException as exc:
        elapsed = time.time() - t0
        print(f"  [pilot] FAILED after {elapsed:.1f}s: {exc!r}")
        raise
    elapsed = time.time() - t0
    print(f"  [pilot] solve completed in {elapsed:.1f}s")

    t_arr = np.asarray(g["t_sliding"], dtype=float)
    y_arr = np.asarray(g["y_sliding"], dtype=float)
    n_phys = int(g["n_phys_s"])
    n_base = int(g["n_base_s"])
    v_hist = y_arr[:, n_base:n_phys]

    out = dict(
        t=t_arr, v_hist=v_hist,
        u_inf=np.max(np.abs(y_arr[:, :n_base]), axis=1),
        v_inf=np.max(np.abs(v_hist), axis=1),
        Nu=v_hist.shape[1],
        n_phys=n_phys, n_base=n_base,
        err_l2=float(g["err_l2"]),
        ref_l2=float(g["ref_l2"]),
        slip_phys=np.asarray(g["slip_phys"], dtype=float),
        slip_anal_phys=np.asarray(g["slip_anal_phys"], dtype=float),
        s_param=np.asarray(g["s_param"], dtype=float),
        elapsed=elapsed,
    )
    out["rel_l2"] = out["err_l2"] / max(out["ref_l2"], 1e-30)
    return out


# --------------------------------------------------------------------------- #
# DMD wrapper (reuses sea.dmd)                                                #
# --------------------------------------------------------------------------- #

def _leading_eig_dmd(t: np.ndarray, V: np.ndarray, *,
                     t_start: float, rank: int) -> tuple[complex, np.ndarray]:
    """Leading DMD eigenvalue (max Re λ) and the full sorted spectrum.

    On failure (too few snapshots after t_start) returns NaN.
    """
    try:
        out = sea.dmd(t, V, t_start=t_start, rank=rank)
        return complex(out["lam"][0]), np.asarray(out["lam"])
    except Exception as exc:
        print(f"  [dmd] failed: {exc}")
        return complex("nan+nanj"), np.array([], dtype=complex)


# --------------------------------------------------------------------------- #
# CSV bookkeeping                                                             #
# --------------------------------------------------------------------------- #

CSV_FIELDS = [
    "N", "lumped_coupling", "taylor_method", "include_taylor",
    "tmax_phys",
    "status",
    "u_inf_final", "err_l2", "ref_l2", "rel_l2",
    "leading_re_lambda", "leading_im_lambda",
    "elapsed_s", "npz_path", "error",
]


def _load_done(csv_path: pathlib.Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (int(row["N"]), row["lumped_coupling"],
                   row["taylor_method"], row["include_taylor"])
            done.add(key)
    return done


def _append_row(csv_path: pathlib.Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def _config_label(N, lumped, method, taylor) -> str:
    return (f"N{N}_L{str(lumped)}_M{method}_T{int(bool(taylor))}")


def _run_one(*, N: int, lumped: Any, method: str, taylor: bool,
             tmax: float, t_linear_start: float, dmd_rank: int,
             out_dir: pathlib.Path) -> dict:
    label = _config_label(N, lumped, method, taylor)
    npz_path = out_dir / f"sweep_{label}_TMAX{int(tmax)}.npz"

    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(N=N, lumped_coupling=str(lumped), taylor_method=method,
               include_taylor=str(bool(taylor)), tmax_phys=tmax,
               npz_path=str(npz_path))

    try:
        traj = _run_pilot_with_l2(N=N, tmax=tmax, taylor=taylor,
                                   taylor_method=method,
                                   lumped_coupling=lumped)
    except BaseException as exc:
        row["status"] = "PILOT_FAILED"
        row["error"] = repr(exc)[:200]
        return row

    leading, all_eigs = _leading_eig_dmd(
        traj["t"], traj["v_hist"], t_start=t_linear_start, rank=dmd_rank)

    np.savez(npz_path,
             t=traj["t"], v_hist=traj["v_hist"],
             u_inf=traj["u_inf"], v_inf=traj["v_inf"],
             slip_phys=traj["slip_phys"],
             slip_anal_phys=traj["slip_anal_phys"],
             s_param=traj["s_param"],
             err_l2=traj["err_l2"], ref_l2=traj["ref_l2"],
             rel_l2=traj["rel_l2"],
             dmd_eigvals=all_eigs)

    u_final = float(traj["u_inf"][-1])
    blew_up = (not np.isfinite(u_final)) or u_final > 1e3
    if blew_up:
        row["status"] = "BLEW_UP"
    elif np.isfinite(leading.real) and leading.real > 1e-6:
        row["status"] = "UNSTABLE"
    else:
        row["status"] = "OK"

    row.update(
        u_inf_final=f"{u_final:.6e}",
        err_l2=f"{traj['err_l2']:.6e}",
        ref_l2=f"{traj['ref_l2']:.6e}",
        rel_l2=f"{traj['rel_l2']:.6f}",
        leading_re_lambda=f"{leading.real:+.6e}",
        leading_im_lambda=f"{leading.imag:+.6e}",
        elapsed_s=f"{traj['elapsed']:.1f}",
    )
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, nargs="*", default=[30, 40, 60])
    p.add_argument("--lumped", type=str, nargs="*",
                   default=["True", "consistent"],
                   choices=list(_LUMPED_VALUES.keys()))
    p.add_argument("--method", type=str, nargs="*",
                   default=["nodal", "l2_project"],
                   choices=["nodal", "l2_project"])
    p.add_argument("--taylor", type=str, nargs="*",
                   default=["True", "False"],
                   choices=["True", "False"])
    p.add_argument("--tmax", type=float, default=30.0)
    p.add_argument("--t-linear-start", type=float, default=8.0,
                   help="DMD post-ramp window start (nondim time)")
    p.add_argument("--dmd-rank", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                   default="/tmp/long_time_convergence_sweep")
    p.add_argument("--csv-name", type=str,
                   default="long_time_convergence_sweep.csv")
    p.add_argument("--force", action="store_true",
                   help="re-run configurations already in the CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name

    done = set() if args.force else _load_done(csv_path)
    print(f"Sweep target dir: {out_dir}")
    print(f"CSV path:         {csv_path}")
    print(f"Already complete: {len(done)} rows")

    configs = [(N, _LUMPED_VALUES[L], M, T == "True")
               for N in args.N
               for L in args.lumped
               for M in args.method
               for T in args.taylor]
    total = len(configs)
    print(f"Planned configs:  {total}")
    print()

    n_run = n_skip = n_fail = 0
    t_overall = time.time()
    for i, (N, lumped, method, taylor) in enumerate(configs, start=1):
        key = (N, str(lumped), method, str(bool(taylor)))
        if key in done:
            print(f"[{i}/{total}] SKIP (already done): {key}")
            n_skip += 1
            continue
        print(f"[{i}/{total}] {key}")
        try:
            row = _run_one(N=N, lumped=lumped, method=method, taylor=taylor,
                           tmax=args.tmax,
                           t_linear_start=args.t_linear_start,
                           dmd_rank=args.dmd_rank, out_dir=out_dir)
        except BaseException:  # last-ditch — keep the sweep alive
            traceback.print_exc()
            row = dict.fromkeys(CSV_FIELDS, "")
            row.update(N=N, lumped_coupling=str(lumped),
                       taylor_method=method,
                       include_taylor=str(bool(taylor)),
                       tmax_phys=args.tmax,
                       status="DRIVER_CRASH",
                       error=traceback.format_exc().splitlines()[-1][:200])
        _append_row(csv_path, row)
        n_run += 1
        if row.get("status") not in ("OK",):
            n_fail += 1
        print(f"   -> status={row.get('status')!r}  rel_l2={row.get('rel_l2')!r}  "
              f"Re(λ)={row.get('leading_re_lambda')!r}\n")

    elapsed = time.time() - t_overall
    print(f"Sweep done. ran={n_run}, skipped={n_skip}, "
          f"non-OK={n_fail}, total wall time {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
