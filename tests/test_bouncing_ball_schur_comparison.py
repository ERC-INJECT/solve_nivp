"""Bouncing ball: analytical vs BE-Schur vs RadauIIA-Schur comparison plot."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from solve_nivp.ncp_contact import build_ncp_contact_blocked
from solve_nivp.integrations import BackwardEulerSchur, RadauIIASchur


def _bouncing_ball_block_system():
    mass = 1.0
    g_acc = 9.81
    gravity = np.array([0.0, -g_acc])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y):
        return np.concatenate([mass * gravity, y[:2]])

    def gap_func(y, t):
        return np.array([y[3]])

    drop_height = 0.5
    y0 = np.array([0.0, 0.0, 0.0, drop_height])
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.0)]

    bs = build_ncp_contact_blocked(
        A=A, rhs_smooth=rhs, y0=y0, contacts=contacts, gap_func=gap_func,
    )
    return bs, y0, drop_height, g_acc


def _analytical_bouncing_ball(t_arr, drop_height, g):
    """Perfectly inelastic (e=0) bouncing ball: free-fall then rest on floor."""
    t_impact = np.sqrt(2.0 * drop_height / g)
    q = np.empty_like(t_arr)
    v = np.empty_like(t_arr)
    for i, t in enumerate(t_arr):
        if t <= t_impact:
            q[i] = drop_height - 0.5 * g * t ** 2
            v[i] = -g * t
        else:
            q[i] = 0.0
            v[i] = 0.0
    return q, v


def run_comparison():
    bs, y0_phys, drop_height, g_acc = _bouncing_ball_block_system()
    n_p = bs.n_phys
    n_aug = n_p + bs.n_react

    h = 0.005
    t_end = 0.6
    n_steps = int(round(t_end / h))

    be_opts = {"maxiter": 30, "tol": 1e-10}
    radau_opts = {"maxiter": 60, "tol": 1e-10}

    results = {}
    for label, integrator in [
        ("BE (Schur)", BackwardEulerSchur(
            A=bs._cs.A, schur_solver_opts=be_opts)),
        ("Radau IIA s=2 (Schur)", RadauIIASchur(
            stages=2, A=bs._cs.A, schur_solver_opts=radau_opts)),
    ]:
        t_hist = [0.0]
        y_cur = np.zeros(n_aug)
        y_cur[:n_p] = y0_phys
        qy_hist = [y_cur[3]]
        vy_hist = [y_cur[1]]
        rn_hist = [0.0]

        t = 0.0
        for _ in range(n_steps):
            y_new, _, err, ok, iters = integrator.step(bs, t, y_cur, h)
            assert ok, f"{label} failed at t={t:.4f}, err={err}"
            t += h
            y_cur = y_new
            t_hist.append(t)
            qy_hist.append(y_cur[3])
            vy_hist.append(y_cur[1])
            rn_hist.append(y_cur[n_p] if n_aug > n_p else 0.0)

        results[label] = {
            "t": np.array(t_hist),
            "qy": np.array(qy_hist),
            "vy": np.array(vy_hist),
            "rn": np.array(rn_hist),
        }

    t_fine = np.linspace(0, t_end, 2000)
    q_an, v_an = _analytical_bouncing_ball(t_fine, drop_height, g_acc)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    ax = axes[0]
    ax.plot(t_fine, q_an, "k-", lw=1.5, label="Analytical")
    for label, d in results.items():
        ax.plot(d["t"], d["qy"], "o-", ms=2, lw=0.8, label=label)
    ax.set_ylabel("Height $q_y$")
    ax.legend(fontsize=8)
    ax.set_title(f"Bouncing ball (e=0), h = {h}")

    ax = axes[1]
    ax.plot(t_fine, v_an, "k-", lw=1.5, label="Analytical")
    for label, d in results.items():
        ax.plot(d["t"], d["vy"], "o-", ms=2, lw=0.8, label=label)
    ax.set_ylabel("Velocity $v_y$")
    ax.legend(fontsize=8)

    t_impact = np.sqrt(2 * drop_height / g_acc)
    r_steady = 1.0 * g_acc

    ax = axes[2]
    ax.axhline(r_steady, color="k", ls="--", lw=0.8, label=f"$mg$ = {r_steady:.2f}")
    ax.axvline(t_impact, color="gray", ls=":", lw=0.7, label=f"$t_{{impact}}$")
    for label, d in results.items():
        ax.plot(d["t"], d["rn"], "o-", ms=2, lw=0.8, label=label)
    ax.set_ylabel("$r_n$ (full scale)")
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.axhline(r_steady, color="k", ls="--", lw=0.8, label=f"$mg$ = {r_steady:.2f}")
    ax.axvline(t_impact, color="gray", ls=":", lw=0.7, label=f"$t_{{impact}}$")
    for label, d in results.items():
        ax.plot(d["t"], d["rn"], "o-", ms=2, lw=0.8, label=label)
    ax.set_ylim(-1, 3 * r_steady)
    ax.set_ylabel("$r_n$ (zoomed)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "images" / "bouncing_ball_schur_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")
    return out


def test_bouncing_ball_schur_comparison_plot():
    """Generate comparison plot (analytical vs BE vs Radau s=2)."""
    out = run_comparison()
    assert out.exists()


if __name__ == "__main__":
    run_comparison()
