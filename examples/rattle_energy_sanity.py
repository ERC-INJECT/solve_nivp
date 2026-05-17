"""Rigid 1-DOF bouncing ball — RATTLE energy-conservation sanity test.

Point mass under gravity, rigid floor at q = 0, Newton restitution e = 1.
For Moreau's impact law with e = 1 the total mechanical energy
E = 1/2 m u^2 + m g q must be conserved to integrator tolerance.
If this test leaks energy, the bug is in rattle_contact.py, not in the
FEM discretisation.
"""

import numpy as np
import matplotlib.pyplot as plt

from solve_nivp.rattle_contact import (
    RattleMechanicalSystem,
    RattleContactSpec,
    build_rattle_system,
    solve_dynamic_rattle_contact,
)


def run(e: float, n_steps: int = 4000, t_end: float = 4.0):
    m = 1.0
    g = 9.81
    q0 = np.array([1.0])
    u0 = np.array([0.0])

    mech = RattleMechanicalSystem(
        nq=1, nu=1, q0=q0, u0=u0,
        M=m * np.eye(1),
        h_force=lambda t, q, u: np.array([-m * g]),
    )

    contact = RattleContactSpec(
        g_N=lambda t, q: float(q[0]),
        W_N=np.array([1.0]),
        gamma_F=lambda t, q, u: np.zeros(1),
        W_F=np.zeros((1, 1)),
        mu=0.0,
        e=e,
        n_F=1,
    )

    system = build_rattle_system(mech, contacts=[contact])
    result = solve_dynamic_rattle_contact(
        system, (0.0, t_end), n_steps=n_steps,
    )

    t = result.times
    q = result.states[:, 0]
    u = result.states[:, 1]
    KE = 0.5 * m * u ** 2
    PE = m * g * q
    E = KE + PE
    return t, q, u, E


def refinement_study():
    print("\nStep-refinement study for e = 1.0 (4 s, 4 bounces):")
    for n in (1000, 2000, 4000, 8000, 16000, 32000):
        t, q, u, E = run(e=1.0, n_steps=n)
        drift = (E[-1] - E[0]) / E[0]
        peak = np.max(np.abs((E - E[0]) / E[0]))
        print(f"  n_steps = {n:6d}  h = {4.0/n:.2e}  "
              f"E(end)/E(0)-1 = {drift:+.3e}  peak = {peak:.3e}")


def main():
    refinement_study()
    for e in (1.0, 0.0):
        t, q, u, E = run(e=e)
        E0 = E[0]
        drift = (E - E0) / E0
        print(f"e = {e}:  E(0) = {E0:.6e}  "
              f"max |dE/E| = {np.max(np.abs(drift)):.3e}  "
              f"E(end)/E(0) = {E[-1] / E0:.6f}")

        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax[0].plot(t, q)
        ax[0].set_ylabel("q")
        ax[0].set_title(f"RATTLE bouncing point mass, e = {e}")
        ax[1].plot(t, E)
        ax[1].axhline(E0, color="k", ls="--", lw=0.8)
        ax[1].set_ylabel("E_total")
        ax[1].set_xlabel("t")
        fig.tight_layout()
        fig.savefig(f"rattle_energy_sanity_e{e}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
