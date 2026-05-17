"""Bouncing-block FEM benchmark — RATTLE symplectic backend, fixed step.

Generates:
  - results/geometry_{nx}x{ny}.png
  - results/bounce_{nx}x{ny}.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.collections import PolyCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_nivp
from problem import build
from adapter import as_rattle, as_ncp, as_desaxce
from solve_nivp.rattle_contact import solve_dynamic_rattle_contact

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_geometry(problem, path: Path):
    mesh = problem.mesh
    p = mesh.p
    t = mesh.t

    fig, ax = plt.subplots(figsize=(8, 5))
    for cell in t.T:
        xs = p[0, cell]
        ys = p[1, cell]
        xs_c = np.append(xs, xs[0])
        ys_c = np.append(ys, ys[0])
        ax.plot(xs_c, ys_c, color="C0", lw=0.6, alpha=0.7)

    ax.plot(problem.bottom_node_x, problem.bottom_node_y_ref,
            "o", color="C3", markersize=6, label="contact nodes")

    xmin = p[0].min() - 0.2 * problem.L
    xmax = p[0].max() + 0.2 * problem.L
    ax.axhline(0.0, color="k", lw=2, label="rigid floor y=0")
    ax.fill_between([xmin, xmax], -0.15, 0.0, color="k", alpha=0.15, hatch="//")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.2, problem.drop_height + problem.H + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"bouncing block  n_nodes={problem.n_nodes}  "
        f"n_contacts={len(problem.contacts)}  drop_height={problem.drop_height}"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeseries(t_arr, q_arr, u_arr, problem, wall, steps, path: Path):
    n_dof = problem.n_dof
    vx = u_arr[:, 0::2]
    vy = u_arr[:, 1::2]
    mean_vx = vx.mean(axis=1)
    mean_vy = vy.mean(axis=1)

    uy_idx = 2 * problem.bottom_nodes + 1
    ux_idx = 2 * problem.bottom_nodes
    u_y_bot = q_arr[:, uy_idx]
    u_x_bot = q_arr[:, ux_idx]
    gap = problem.bottom_node_y_ref[None, :] + u_y_bot
    min_gap = gap.min(axis=1)
    max_slip = np.max(np.abs(u_x_bot - u_x_bot[0]), axis=1)

    M_diag = problem.M.diagonal()
    m_nodal_x = M_diag[0::2]
    m_nodal_y = M_diag[1::2]
    m_tot = 0.5 * M_diag.sum()

    T_total = 0.5 * (u_arr ** 2 * M_diag[None, :]).sum(axis=1)
    v_com_x = (u_arr[:, 0::2] * m_nodal_x[None, :]).sum(axis=1) / m_tot
    v_com_y = (u_arr[:, 1::2] * m_nodal_y[None, :]).sum(axis=1) / m_tot
    T_com = 0.5 * m_tot * (v_com_x ** 2 + v_com_y ** 2)
    T_int = T_total - T_com

    Ku = (problem.K @ q_arr.T).T
    W_el = 0.5 * (q_arr * Ku).sum(axis=1)
    node_y = problem.mesh.p[1, :]
    PE_g = problem.g * (m_nodal_y * (node_y + q_arr[:, 1::2])).sum(axis=1)
    E_tot = T_total + W_el + PE_g

    E0 = E_tot[0]
    T_com0 = T_com[0]
    drift = (E_tot - E0) / abs(E0)
    print(f"[energy] E_tot(0)={E0:.6e}  T_com(0)={T_com0:.6e}  m_tot={m_tot:.4f}")
    print(f"[energy] max |dE_tot/E0|={np.max(np.abs(drift)):+.3e}  "
          f"E_tot(end)/E_tot(0)={E_tot[-1] / E0:.6f}")
    print(f"[energy] T_com(end)/T_com(0)={T_com[-1] / T_com0 if T_com0 > 0 else 0:.6f}  "
          f"max T_int={T_int.max():.3e}")

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes[0, 0].plot(t_arr, min_gap, "C0")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_ylabel("min gap")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t_arr, max_slip, "C1")
    axes[0, 1].set_ylabel("max |tangential disp|")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t_arr, mean_vy, "C2")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].set_ylabel("mean vy")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t_arr, mean_vx, "C3")
    axes[1, 1].set_ylabel("mean vx")
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(t_arr, T_com, label=r"$T_{\mathrm{COM}}$")
    axes[2, 0].plot(t_arr, T_int, label=r"$T_{\mathrm{int}}$")
    axes[2, 0].plot(t_arr, W_el, label=r"$W_{\mathrm{el}}$")
    axes[2, 0].plot(t_arr, E_tot - E_tot[0] + T_com[0], "k--",
                    label=r"$E_{\mathrm{tot}}$ (shifted)")
    axes[2, 0].set_ylabel("energy")
    axes[2, 0].set_xlabel("t")
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend(fontsize=8, ncol=2)

    axes[2, 1].plot(t_arr, drift, "C3", label=r"$(E_{\mathrm{tot}}-E_0)/E_0$")
    axes[2, 1].axhline(0, color="k", lw=0.5)
    axes[2, 1].set_ylabel("total-energy drift")
    axes[2, 1].set_xlabel("t")
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].legend(fontsize=8)

    fig.suptitle(
        f"bouncing block RATTLE  "
        f"n_phys={problem.n_phys}  n_contacts={len(problem.contacts)}  "
        f"wall={wall:.2f}s  steps={steps}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def animate_mesh(t_arr, q_arr, problem, path: Path, label: str,
                 n_frames: int = 120):
    mesh = problem.mesh
    p0 = mesh.p.copy()
    cells = mesh.t.T  # (n_elem, 4)

    n_steps = q_arr.shape[0]
    frame_idx = np.linspace(0, n_steps - 1, min(n_frames, n_steps)).astype(int)

    # Deformed positions per frame
    def nodes_at(k):
        u = q_arr[k]
        xs = p0[0] + u[0::2]
        ys = p0[1] + u[1::2]
        return xs, ys

    xs_all = []
    ys_all = []
    for k in frame_idx:
        xs, ys = nodes_at(k)
        xs_all.append(xs)
        ys_all.append(ys)
    xs_all = np.asarray(xs_all)
    ys_all = np.asarray(ys_all)

    xmin = xs_all.min() - 0.2 * problem.L
    xmax = xs_all.max() + 0.2 * problem.L
    ymin = min(-0.15, ys_all.min() - 0.1)
    ymax = ys_all.max() + 0.15

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0.0, color="k", lw=2)
    ax.fill_between([xmin, xmax], ymin, 0.0, color="k", alpha=0.15, hatch="//")

    verts0 = [np.column_stack([xs_all[0][c], ys_all[0][c]]) for c in cells]
    pc = PolyCollection(verts0, facecolors="C0", edgecolors="k",
                        linewidths=0.5, alpha=0.6)
    ax.add_collection(pc)
    title = ax.set_title("")

    def update(frame):
        xs = xs_all[frame]
        ys = ys_all[frame]
        verts = [np.column_stack([xs[c], ys[c]]) for c in cells]
        pc.set_verts(verts)
        title.set_text(f"{label}  t={t_arr[frame_idx[frame]]:.3f}")
        return pc, title

    anim = manim.FuncAnimation(
        fig, update, frames=len(frame_idx), interval=40, blit=False
    )
    anim.save(path, writer=manim.PillowWriter(fps=25))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument(
        "--backend",
        choices=["rattle", "ncp_radau", "ncp_be",
                 "desaxce_radau", "desaxce_be"],
        default="rattle",
    )
    args = ap.parse_args()

    problem = build(nx=args.nx, ny=args.ny)
    print(f"Problem: nx={args.nx}, ny={args.ny}, n_phys={problem.n_phys}, "
          f"n_contacts={len(problem.contacts)}")

    geo_path = RESULTS_DIR / f"geometry_{args.nx}x{args.ny}.png"
    plot_geometry(problem, geo_path)
    print(f"Saved {geo_path}")

    if args.backend == "rattle":
        bundle = as_rattle(problem)
        solver_opts = dict(
            newton_tol=1.0e-6,
            newton_max_iter=100,
            fixed_point_tol=1.0e-6,
            fixed_point_max_iter=200,
            linesearch_max_iter=10,
            stage1_method="semismooth_newton",
        )
        start = time.time()
        result = solve_dynamic_rattle_contact(
            bundle.system,
            (0.0, args.t_end),
            n_steps=args.n_steps,
            solver_opts=solver_opts,
        )
        wall = time.time() - start
        print(f"[rattle] wall={wall:.2f}s  steps={result.times.shape[0]}")
        if result.failure is not None:
            print(f"[rattle] FAILURE: {result.failure}")
        t_arr = result.times
        # RATTLE state layout: [q, u]
        q_arr = result.states[:, :problem.n_dof]
        u_arr = result.states[:, problem.n_dof:2 * problem.n_dof]
        label = "RATTLE"
    else:
        if args.backend.startswith("ncp"):
            bundle = as_ncp(problem)
            tag = "NCP"
        else:
            bundle = as_desaxce(problem)
            tag = "DeSaxce"
        method = "radau_iia" if args.backend.endswith("radau") else "backward_euler"
        integrator_opts = dict(stages=2) if method == "radau_iia" else {}
        integrator_opts.update(getattr(bundle, "integrator_opts", {}) or {})
        h_fixed = (args.t_end - 0.0) / args.n_steps
        start = time.time()
        t, y, h, fk, info = solve_nivp.solve_nivp(
            fun=bundle.fun,
            t_span=(0.0, args.t_end),
            y0=bundle.y0,
            method=method,
            projection=bundle.projection,
            solver="semismooth_newton",
            projection_opts=bundle.projection_opts,
            solver_opts=bundle.solver_opts,
            adaptive=False,
            h0=h_fixed,
            component_slices=bundle.component_slices,
            A=bundle.A,
            verbose=False,
            integrator_opts=integrator_opts,
        )
        wall = time.time() - start
        print(f"[{args.backend}] wall={wall:.2f}s  steps={len(t)}  t_end={t[-1]:.4f}")
        if isinstance(info, (list, tuple)) and len(info) > 0:
            last = info[-1]
            print(f"[{args.backend}] last info entry: {last}")
        t_arr = np.asarray(t)
        n_dof = problem.n_dof
        u_arr = y[:, :n_dof]
        q_arr = y[:, n_dof:2 * n_dof]
        label = f"{tag}-{method}"

    ts_path = RESULTS_DIR / f"bounce_{args.nx}x{args.ny}_{args.backend}.png"
    plot_timeseries(t_arr, q_arr, u_arr, problem, wall,
                    t_arr.shape[0], ts_path)
    print(f"Saved {ts_path}")

    gif_path = RESULTS_DIR / f"bounce_{args.nx}x{args.ny}_{args.backend}.gif"
    animate_mesh(t_arr, q_arr, problem, gif_path, label=label)
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
