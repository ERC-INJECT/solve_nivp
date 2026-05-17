"""Tests for the theta-Moreau-Jean-Fremond integrator."""

import numpy as np
import pytest

from solve_nivp.moreau_jean_fremond import (
    build_moreau_jean_fremond,
    MoreauJeanFremondStepper,
)


def _no_contact_stepper(M, K, C, n_solid, F_callable=None, theta=0.5):
    """Stepper with zero contacts -- tests the smooth path only."""
    if F_callable is None:
        F_callable = lambda t: np.zeros(n_solid)
    stepper = MoreauJeanFremondStepper(
        M=M, K=K, C=C,
        block_slices=[], e_N_vec=np.zeros(0),
        H_callable=np.zeros((0, n_solid)),
        F_callable=F_callable,
        theta=theta,
        aux_law=lambda aux, u, sl, h, p: dict(aux),
    )
    return stepper


def test_smooth_undamped_oscillator_energy_at_theta_half():
    # m = 1, k = 4, no damping. Period = pi. Energy must be exactly conserved
    # at theta = 1/2 (Crank-Nicolson is symplectic for harmonic oscillators).
    M = np.array([[1.0]])
    K = np.array([[4.0]])
    C = np.array([[0.0]])
    stepper = _no_contact_stepper(M, K, C, 1, theta=0.5)

    y = np.array([1.0, 0.0])  # q=1, v=0
    aux = {}
    h = 0.01
    energies = []
    for _ in range(200):
        y, aux, info = stepper.step(0.0, y, aux, h, return_diagnostic=True)
        E = 0.5 * y[1]**2 + 0.5 * 4.0 * y[0]**2
        energies.append(E)
    energies = np.array(energies)
    # Crank-Nicolson conserves a quadratic form for linear systems (only true
    # for the modified energy I + h^2/4 K, not exactly E itself, but the drift
    # is bounded and not growing). Check no secular drift.
    assert energies.max() - energies.min() < 1.0e-3


def test_smooth_damped_oscillator_dissipates_energy():
    M = np.array([[1.0]])
    K = np.array([[4.0]])
    C = np.array([[0.5]])
    stepper = _no_contact_stepper(M, K, C, 1, theta=0.5)

    y = np.array([1.0, 0.0])
    aux = {}
    h = 0.01
    E0 = 0.5 * y[1]**2 + 0.5 * 4.0 * y[0]**2
    for _ in range(200):
        y, aux, info = stepper.step(0.0, y, aux, h)
    E_final = 0.5 * y[1]**2 + 0.5 * 4.0 * y[0]**2
    assert E_final < 0.5 * E0  # damping has bled off energy


def test_smooth_constant_force_drift():
    # F = const, no stiffness. v(t) = F/m * t.
    M = np.array([[2.0]])
    K = np.array([[0.0]])
    C = np.array([[0.0]])
    F0 = 1.5
    stepper = _no_contact_stepper(
        M, K, C, 1, F_callable=lambda t: np.array([F0]), theta=0.5,
    )
    y = np.array([0.0, 0.0])
    h = 0.01
    aux = {}
    for k in range(100):
        y, aux, _ = stepper.step(k * h, y, aux, h)
    expected_v = F0 / 2.0 * (100 * h)
    np.testing.assert_allclose(y[1], expected_v, rtol=1.0e-9)


def test_inelastic_impact_eN_zero():
    # 1D: mass 1 falling at v=-2, gravity g=9.81. After impact at e=0
    # the velocity should be 0 (sticking).
    mass = 1.0
    g = 9.81
    M = np.array([[mass]])
    K = np.array([[0.0]])
    C = np.array([[0.0]])
    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 1, "mu_init": 0.0}],
        H_callable=np.array([[1.0]]),
        F_callable=lambda t: np.array([-mass * g]),
        e_N=0.0, theta=0.5,
    )
    y = np.array([0.0, -2.0])
    h = 0.01
    y_new, aux_new, info = stepper.step(0.0, y, aux, h, return_diagnostic=True)
    # Post-impact velocity should satisfy contact: u_N = v_new >= 0,
    # complementarity v_new * p_contact = 0.  For an inelastic impact, v_new=0.
    np.testing.assert_allclose(y_new[1], 0.0, atol=1.0e-9)
    # Slack non-negative (energy dissipated by inelastic impact).
    assert info["slack"] >= -1.0e-12


def test_elastic_impact_eN_one():
    # e = 1, post-impact velocity = +|pre|.
    mass = 1.0
    g = 0.0  # ignore gravity for clean check
    M = np.array([[mass]])
    K = np.array([[0.0]])
    C = np.array([[0.0]])
    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 1, "mu_init": 0.0}],
        H_callable=np.array([[1.0]]),
        F_callable=lambda t: np.array([-mass * g]),
        e_N=1.0, theta=0.5,
    )
    y = np.array([0.0, -3.0])
    h = 0.01
    y_new, _, info = stepper.step(0.0, y, aux, h, return_diagnostic=True)
    np.testing.assert_allclose(y_new[1], 3.0, atol=1.0e-9)


def test_partial_restitution_eN_half():
    mass = 1.0
    M = np.array([[mass]])
    K = np.array([[0.0]])
    C = np.array([[0.0]])
    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 1, "mu_init": 0.0}],
        H_callable=np.array([[1.0]]),
        F_callable=lambda t: np.array([0.0]),
        e_N=0.5, theta=0.5,
    )
    y = np.array([0.0, -2.0])
    y_new, _, _ = stepper.step(0.0, y, aux, 0.01)
    np.testing.assert_allclose(y_new[1], 1.0, atol=1.0e-9)


def test_invalid_theta_with_high_restitution_raises():
    # theta = 1 with e = 0.5 violates Acary-Collins-Craft Prop 1.
    M = np.eye(1); K = np.zeros((1, 1)); C = np.zeros((1, 1))
    with pytest.raises(ValueError, match="theta"):
        build_moreau_jean_fremond(
            M, K, C,
            contacts=[{"block_size": 1, "mu_init": 0.0}],
            H_callable=np.array([[1.0]]),
            F_callable=lambda t: np.zeros(1),
            e_N=0.5, theta=1.0,
        )


def test_2d_sliding_block_constant_drive_reaches_coulomb_limit():
    # 2D block on a horizontal surface, constant tangential drive F_T.
    # Mass M=1; K=0; C=0; e=0; mu=0.4.
    # F = (-g, F_T) (gravity normal, drive tangential).
    # Coulomb friction is mu*N = mu * m * g.
    # If F_T > mu*m*g, the block slides; if F_T < mu*m*g, it sticks.
    mass = 1.0
    g = 9.81
    mu = 0.4
    F_T = 0.5 * mu * mass * g  # below threshold: should stick
    M = mass * np.eye(2)
    K = np.zeros((2, 2))
    C = np.zeros((2, 2))

    # H: u = (v_normal, v_tangential) = (v[0], v[1])
    H = np.eye(2)

    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 2, "mu_init": mu}],
        H_callable=H,
        F_callable=lambda t: np.array([-mass * g, F_T]),
        e_N=0.0, theta=0.5,
    )

    # Initial state: at rest at the contact (q[0] = 0, v = 0).
    y = np.array([0.0, 0.0, 0.0, 0.0])
    h = 0.001

    for k in range(50):
        y, aux, info = stepper.step(k * h, y, aux, h)
    # Should stick: tangential velocity stays near zero.
    assert abs(y[3]) < 1.0e-6


def test_2d_sliding_block_above_threshold_slides():
    mass = 1.0
    g = 9.81
    mu = 0.4
    F_T = 1.5 * mu * mass * g  # above threshold: slides
    M = mass * np.eye(2)
    K = np.zeros((2, 2))
    C = np.zeros((2, 2))
    H = np.eye(2)

    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 2, "mu_init": mu}],
        H_callable=H,
        F_callable=lambda t: np.array([-mass * g, F_T]),
        e_N=0.0, theta=0.5,
    )

    y = np.array([0.0, 0.0, 0.0, 0.0])
    h = 0.001
    for k in range(50):
        y, aux, info = stepper.step(k * h, y, aux, h)

    # Tangential velocity grows linearly with effective force F_T - mu*m*g.
    expected_acceleration = (F_T - mu * mass * g) / mass
    expected_v = expected_acceleration * 50 * h
    np.testing.assert_allclose(y[3], expected_v, rtol=2.0e-2)
    # Cone admissibility on the impulse.
    p_contact = info["p_contact"]
    # ||p_T|| = mu * p_N at slip.
    assert abs(np.linalg.norm(p_contact[1:]) - mu * p_contact[0]) < 1.0e-6


def test_slip_weakening_reduces_friction_with_slip():
    # Apply tangential drive; cumulative slip should reduce mu from
    # mu_s towards mu_d as cumulative slip exceeds D_c.
    mass = 1.0
    g = 9.81
    mu_s = 0.6
    mu_d = 0.3
    D_c = 0.001
    F_T = 1.2 * mu_s * mass * g  # above threshold

    M = mass * np.eye(2); K = np.zeros((2, 2)); C = np.zeros((2, 2))
    H = np.eye(2)
    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 2, "mu_init": mu_s}],
        H_callable=H,
        F_callable=lambda t: np.array([-mass * g, F_T]),
        e_N=0.0, theta=0.5,
        aux_law="slip_weakening",
        aux_law_params={"mu_s": mu_s, "mu_d": mu_d, "D_c": D_c},
    )

    y = np.array([0.0, 0.0, 0.0, 0.0])
    h = 0.001
    mu_history = [aux["mu"][0]]
    for k in range(200):
        y, aux, _ = stepper.step(k * h, y, aux, h)
        mu_history.append(aux["mu"][0])

    mu_history = np.array(mu_history)
    # Friction has weakened toward mu_d as slip accumulates.
    assert mu_history[0] == mu_s
    assert mu_history[-1] < mu_history[0]
    assert abs(mu_history[-1] - mu_d) < 0.05


def test_porodynamics_1d_consolidation_no_contact():
    # 1D Biot consolidation without contact: smooth block solve only.
    # Verify the augmented operator handles fluid-pressure coupling.
    n_solid = 2
    n_fluid = 2
    M = np.eye(n_solid)
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    C = np.zeros((n_solid, n_solid))
    S = 0.5 * np.eye(n_fluid)
    D = np.array([[2.0, -1.0], [-1.0, 2.0]])  # discrete Laplacian
    B = 0.1 * np.eye(n_fluid, n_solid)  # weak Biot coupling

    stepper = MoreauJeanFremondStepper(
        M=M, K=K, C=C, S=S, D=D, B_biot=B,
        block_slices=[], e_N_vec=np.zeros(0),
        H_callable=np.zeros((0, n_solid)),
        F_callable=lambda t: np.zeros(n_solid),
        source_callable=lambda t: np.array([0.1, 0.0]),  # source at node 0
        theta=0.5,
        aux_law=lambda aux, u, sl, h, p: dict(aux),
    )

    y = np.zeros(2 * n_solid + n_fluid)
    aux = {}
    h = 0.01
    for k in range(50):
        y, aux, _ = stepper.step(k * h, y, aux, h)
    # Pressure field should have evolved (source pumps fluid in).
    p_pore = y[2 * n_solid:]
    assert p_pore[0] > 0.0  # source node has positive pressure
    # Solid responds via Biot coupling.
    assert np.linalg.norm(y[:n_solid]) > 0.0


def test_warm_start_p_contact_persists_across_steps():
    # Verify aux carries p_contact_prev and stepper warm-starts the SOCCP.
    mass = 1.0
    g = 9.81
    mu = 0.4
    M = mass * np.eye(2); K = np.zeros((2, 2)); C = np.zeros((2, 2))
    H = np.eye(2)

    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 2, "mu_init": mu}],
        H_callable=H,
        F_callable=lambda t: np.array([-mass * g, 0.0]),
        e_N=0.0, theta=0.5,
    )
    y = np.array([0.0, 0.0, 0.0, 0.0])
    h = 0.001
    y, aux, info1 = stepper.step(0.0, y, aux, h)
    iters1 = info1["soccp_outer_iters"]
    y, aux, info2 = stepper.step(h, y, aux, h)
    iters2 = info2["soccp_outer_iters"]
    assert iters2 <= iters1  # warm-start helps


def test_energy_slack_nonnegative_across_regimes():
    # Spend several steps each in stick / slip / take-off; verify the
    # energy slack remains non-negative (Acary-Collins-Craft Prop 1).
    mass = 1.0; g = 9.81; mu = 0.4
    M = mass * np.eye(2); K = np.zeros((2, 2)); C = np.zeros((2, 2))
    H = np.eye(2)
    F_T_seq = [0.0,                       # stick
               1.5 * mu * mass * g,       # slip
               -2.0 * mass * g]           # take-off

    for F_T in F_T_seq:
        stepper, aux = build_moreau_jean_fremond(
            M, K, C,
            contacts=[{"block_size": 2, "mu_init": mu}],
            H_callable=H,
            F_callable=lambda t, _F=F_T: np.array([-mass * g, _F]),
            e_N=0.0, theta=0.5,
        )
        y = np.array([0.0, 0.0, 0.0, 0.0])
        h = 0.001
        for k in range(20):
            y, aux, info = stepper.step(k * h, y, aux, h, return_diagnostic=True)
            assert info["slack"] >= -1.0e-9, (
                f"F_T={F_T}, step={k}, slack={info['slack']}, "
                f"regime={info['regime']}"
            )
