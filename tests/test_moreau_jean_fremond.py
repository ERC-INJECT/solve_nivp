"""Tests for the theta-Moreau-Jean-Fremond integrator."""

import numpy as np
import pytest

from solve_nivp.moreau_jean_fremond import (
    build_moreau_jean_fremond,
    MoreauJeanFremondStepper,
    DescriptorMoreauJeanFremondStepper,
    solve_mjf_adaptive,
)
from solve_nivp.soccp_pgs import fremond_shift_factory, _shift_jacobian_fd
from solve_nivp.nonlinear_solvers import PETSC_AVAILABLE


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


def test_invalid_contact_solver_raises():
    M = np.eye(1); K = np.zeros((1, 1)); C = np.zeros((1, 1))
    with pytest.raises(ValueError, match="contact_solver"):
        build_moreau_jean_fremond(
            M, K, C,
            contacts=[{"block_size": 1, "mu_init": 0.0}],
            H_callable=np.array([[1.0]]),
            F_callable=lambda t: np.zeros(1),
            e_N=0.0, theta=0.5,
            contact_solver="not-a-solver",
        )


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_petsc_ssn_inelastic_impact_matches_pgs():
    mass = 1.0
    g = 9.81
    M = np.array([[mass]])
    K = np.array([[0.0]])
    C = np.array([[0.0]])
    kwargs = dict(
        contacts=[{"block_size": 1, "mu_init": 0.0}],
        H_callable=np.array([[1.0]]),
        F_callable=lambda t: np.array([-mass * g]),
        e_N=0.0,
        theta=0.5,
    )
    stepper_pgs, aux_pgs = build_moreau_jean_fremond(M, K, C, **kwargs)
    stepper_petsc, aux_petsc = build_moreau_jean_fremond(
        M, K, C,
        contact_solver="petsc_ssn",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        **kwargs,
    )

    y0 = np.array([0.0, -2.0])
    y_pgs, _, info_pgs = stepper_pgs.step(0.0, y0, aux_pgs, 0.01)
    y_petsc, _, info_petsc = stepper_petsc.step(0.0, y0, aux_petsc, 0.01)

    np.testing.assert_allclose(y_petsc, y_pgs, atol=1.0e-9)
    np.testing.assert_allclose(
        info_petsc["p_contact"], info_pgs["p_contact"], atol=1.0e-9,
    )
    assert info_petsc["contact_solver"] == "petsc_ssn"
    assert info_petsc["contact_ssn_converged"]
    assert info_petsc["contact_linear_solver"] == "dense"

    stepper_forced, aux_forced = build_moreau_jean_fremond(
        M, K, C,
        contact_solver="petsc_ssn",
        contact_linear_solver="petsc",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        **kwargs,
    )
    y_forced, _, info_forced = stepper_forced.step(0.0, y0, aux_forced, 0.01)
    np.testing.assert_allclose(y_forced, y_pgs, atol=1.0e-9)
    assert info_forced["contact_linear_solver"] == "petsc"


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_petsc_ssn_sliding_block_matches_pgs():
    mass = 1.0
    g = 9.81
    mu = 0.4
    F_T = 1.5 * mu * mass * g
    M = mass * np.eye(2)
    K = np.zeros((2, 2))
    C = np.zeros((2, 2))
    H = np.eye(2)
    kwargs = dict(
        contacts=[{"block_size": 2, "mu_init": mu}],
        H_callable=H,
        F_callable=lambda t: np.array([-mass * g, F_T]),
        e_N=0.0,
        theta=0.5,
    )
    stepper_pgs, aux_pgs = build_moreau_jean_fremond(M, K, C, **kwargs)
    stepper_petsc, aux_petsc = build_moreau_jean_fremond(
        M, K, C,
        contact_solver="petsc_ssn",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        **kwargs,
    )

    y_pgs = np.array([0.0, 0.0, 0.0, 0.0])
    y_petsc = y_pgs.copy()
    for k in range(3):
        y_pgs, aux_pgs, info_pgs = stepper_pgs.step(k * 0.001, y_pgs, aux_pgs, 0.001)
        y_petsc, aux_petsc, info_petsc = stepper_petsc.step(
            k * 0.001, y_petsc, aux_petsc, 0.001,
        )

    np.testing.assert_allclose(y_petsc, y_pgs, atol=1.0e-8)
    np.testing.assert_allclose(
        info_petsc["p_contact"], info_pgs["p_contact"], atol=1.0e-8,
    )
    assert info_petsc["contact_ssn_converged"]
    assert info_petsc["contact_ssn_residual"] < 1.0e-9


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_petsc_ssn_soc_projection_residual_sliding_block_reaches_coulomb_limit():
    mass = 1.0
    g = 9.81
    mu = 0.4
    F_T = 1.5 * mu * mass * g
    M = mass * np.eye(2)
    K = np.zeros((2, 2))
    C = np.zeros((2, 2))
    stepper, aux = build_moreau_jean_fremond(
        M, K, C,
        contacts=[{"block_size": 2, "mu_init": mu}],
        H_callable=np.eye(2),
        F_callable=lambda t: np.array([-mass * g, F_T]),
        e_N=0.0,
        theta=0.5,
        contact_solver="petsc_ssn",
        contact_residual="soc_projection",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    y = np.zeros(4)
    h = 0.001
    n_steps = 50
    for k in range(n_steps):
        y, aux, info = stepper.step(k * h, y, aux, h)

    expected_acceleration = (F_T - mu * mass * g) / mass
    np.testing.assert_allclose(y[3], expected_acceleration * n_steps * h, rtol=2.0e-2)
    p_contact = info["p_contact"]
    assert abs(np.linalg.norm(p_contact[1:]) - mu * p_contact[0]) < 1.0e-6
    assert info["contact_residual"] == "soc_projection"
    assert info["contact_ssn_converged"]
    assert info["contact_ssn_residual"] < 1.0e-9


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_descriptor_mjf_enforces_dae_constraint_with_contact():
    # Descriptor state y = [q, v, ell], with algebraic constraint ell = q.
    # The contact impulse acts on v, while the theta state must satisfy the
    # algebraic row.  Since the old state is consistent, y_{n+1} is also
    # consistent when the theta state is constrained.
    A = np.diag([1.0, 1.0, 0.0])

    def rhs(t, y):
        return np.array([y[1], 0.0, 0.0])

    def rhs_jac(t, y):
        J = np.zeros((3, 3))
        J[0, 1] = 1.0
        return J

    constraint = {
        "g": lambda q: q.copy(),
        "dg_dy": lambda q: np.eye(1),
        "y_slice": slice(0, 1),
        "q_slice": slice(2, 3),
    }
    stepper = DescriptorMoreauJeanFremondStepper(
        A=A,
        rhs_callable=rhs,
        rhs_jac_callable=rhs_jac,
        D_extract=np.array([[0.0, 1.0, 0.0]]),
        B=np.array([[0.0], [1.0], [0.0]]),
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu": 0.0, "e": 0.0}],
        constraints=[constraint],
        theta=0.5,
        contact_solver="petsc_ssn",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        theta_linear_solver="petsc",
        theta_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    aux = {"mu": np.array([0.0])}
    y0 = np.array([0.0, -1.0, 0.0])

    y1, aux1, info = stepper.step(0.0, y0, aux, 0.1)

    np.testing.assert_allclose(y1[2], y1[0], atol=1.0e-12)
    np.testing.assert_allclose(y1[1], 0.0, atol=1.0e-10)
    assert info["contact_solver"] == "petsc_ssn"
    assert info["contact_ssn_converged"]


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_descriptor_mjf_contact_offset_shifts_cone_without_load():
    A = np.diag([1.0, 1.0, 0.0])

    def rhs(t, y):
        return np.array([y[1], 0.0, 0.0])

    def rhs_jac(t, y):
        J = np.zeros((3, 3))
        J[0, 1] = 1.0
        return J

    constraint = {
        "g": lambda q: q.copy(),
        "dg_dy": lambda q: np.eye(1),
        "y_slice": slice(0, 1),
        "q_slice": slice(2, 3),
    }
    stepper = DescriptorMoreauJeanFremondStepper(
        A=A,
        rhs_callable=rhs,
        rhs_jac_callable=rhs_jac,
        D_extract=np.array([[0.0, 1.0, 0.0]]),
        B=np.array([[0.0], [1.0], [0.0]]),
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu": 0.0, "e": 0.0}],
        constraints=[constraint],
        theta=0.5,
        contact_solver="petsc_ssn",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        theta_linear_solver="petsc",
        theta_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        contact_offset_force=lambda y, t: np.array([4.0]),
    )
    y0 = np.zeros(3)

    y1, aux1, info = stepper.step(0.0, y0, {"mu": np.array([0.0])}, 0.25)

    np.testing.assert_allclose(y1, y0, atol=1.0e-12)
    np.testing.assert_allclose(info["p_contact"], np.zeros(1), atol=1.0e-12)
    np.testing.assert_allclose(info["p_contact_effective"], np.array([1.0]), atol=1.0e-12)


@pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py is not available")
def test_descriptor_mjf_contact_offset_uses_storage_units_for_scaled_reactions():
    A = np.diag([1.0, 1.0, 0.0])

    def rhs(t, y):
        return np.array([y[1], 0.0, 0.0])

    def rhs_jac(t, y):
        J = np.zeros((3, 3))
        J[0, 1] = 1.0
        return J

    constraint = {
        "g": lambda q: q.copy(),
        "dg_dy": lambda q: np.eye(1),
        "y_slice": slice(0, 1),
        "q_slice": slice(2, 3),
    }
    stepper = DescriptorMoreauJeanFremondStepper(
        A=A,
        rhs_callable=rhs,
        rhs_jac_callable=rhs_jac,
        D_extract=np.array([[0.0, 1.0, 0.0]]),
        B=np.array([[0.0], [1.0], [0.0]]),
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu": 0.0, "e": 0.0}],
        constraints=[constraint],
        theta=0.5,
        contact_solver="petsc_ssn",
        contact_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        theta_linear_solver="petsc",
        theta_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        contact_offset_force=lambda y, t: np.array([4.0]),
        reaction_state_to_reported_scale=0.5,
    )

    y1, aux1, info = stepper.step(0.0, np.zeros(3), {"mu": np.array([0.0])}, 0.25)

    np.testing.assert_allclose(y1, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(info["p_contact"], np.zeros(1), atol=1.0e-12)
    np.testing.assert_allclose(info["p_contact_effective"], np.array([2.0]), atol=1.0e-12)
    np.testing.assert_allclose(info["p_contact_offset"], np.array([2.0]), atol=1.0e-12)


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

def _descriptor_sliding_block(mu, F_T, *, mass=1.0, g=9.81, n_slip=0):
    """First-order descriptor sliding block: y = [q_n, q_t, v_n, v_t(, s)].

    With n_slip=1 a cumulative-slip row ds/dt = |v_t| is appended for
    slip-weakening tests.
    """
    n = 4 + n_slip
    A = np.eye(n)

    def rhs(t, y):
        out = np.zeros(n)
        out[0] = y[2]
        out[1] = y[3]
        out[2] = -mass * g
        out[3] = F_T
        if n_slip:
            out[4] = abs(y[3])
        return out

    def rhs_jac(t, y):
        J = np.zeros((n, n))
        J[0, 2] = 1.0
        J[1, 3] = 1.0
        if n_slip:
            J[4, 3] = 1.0 if y[3] >= 0.0 else -1.0
        return J

    D = np.zeros((2, n))
    D[0, 2] = 1.0
    D[1, 3] = 1.0
    B = np.zeros((n, 2))
    B[2, 0] = 1.0
    B[3, 1] = 1.0
    stepper = DescriptorMoreauJeanFremondStepper(
        A=A,
        rhs_callable=rhs,
        rhs_jac_callable=rhs_jac,
        D_extract=D,
        B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu, "e": 0.0}],
        theta=0.5,
        contact_solver="petsc_ssn",
        theta_linear_solver="scipy",
    )
    return stepper


def test_fremond_shift_analytic_jacobian_matches_fd():
    shift = fremond_shift_factory(
        np.array([0.3]), np.array([0.2]), np.array([0.1]), theta=0.5,
    )
    u_slip = np.array([0.2, -0.7])
    np.testing.assert_allclose(
        shift.jacobian(u_slip, 0), _shift_jacobian_fd(shift, u_slip, 0),
        atol=1.0e-6,
    )
    u_kink = np.array([0.2, 0.0])
    J_kink = shift.jacobian(u_kink, 0)
    np.testing.assert_allclose(
        J_kink, _shift_jacobian_fd(shift, u_kink, 0), atol=1.0e-6,
    )
    assert J_kink[0, 1] == pytest.approx(0.3)


def test_descriptor_theta_cache_consistent_across_mu_and_h():
    # One stepper instance reuses the cached theta system across calls with
    # identical (t, y, h) but different mu, then rebuilds for a new h.  Every
    # result must match a fresh stepper with no cache history.
    mass, g = 1.0, 9.81
    F_T = 1.5 * 0.4 * mass * g
    y0 = np.array([0.0, 0.0, 0.0, 0.0])
    h = 0.01
    shared = _descriptor_sliding_block(0.4, F_T)

    cases = [
        ({"mu": np.array([0.4])}, h),
        ({"mu": np.array([0.05])}, h),
        ({"mu": np.array([0.05])}, 0.5 * h),
    ]
    for aux, h_k in cases:
        y_shared, _, info_shared = shared.step(0.0, y0, dict(aux), h_k)
        fresh = _descriptor_sliding_block(0.4, F_T)
        y_fresh, _, info_fresh = fresh.step(0.0, y0, dict(aux), h_k)
        np.testing.assert_allclose(y_shared, y_fresh, atol=1.0e-12)
        np.testing.assert_allclose(
            info_shared["p_contact"], info_fresh["p_contact"], atol=1.0e-12,
        )

    y_a, _, info_a = shared.step(0.0, y0, {"mu": np.array([0.4])}, h)
    y_b, _, info_b = shared.step(0.0, y0, {"mu": np.array([0.05])}, h)
    assert not np.allclose(y_a, y_b)
    assert abs(info_a["p_contact"][1]) > abs(info_b["p_contact"][1])



def test_descriptor_mjf_state_dependent_mu_uses_theta_state_without_outer_fixed_point():
    mass, g = 1.0, 9.81
    mu_static = 0.6
    mu_dynamic = 0.2
    D_c = 1.0e-3
    F_T = 1.5 * mu_static * mass * g
    stepper = _descriptor_sliding_block(mu_static, F_T, mass=mass, g=g, n_slip=1)

    def mu_from_state(z_theta):
        slip = np.asarray(z_theta[4:5], dtype=float)
        return mu_static - (mu_static - mu_dynamic) * np.minimum(slip / D_c, 1.0)

    dmu_calls = {"count": 0}

    def dmu_dstate(z_theta):
        dmu_calls["count"] += 1
        slip = float(z_theta[4])
        grad = np.zeros((1, z_theta.size), dtype=float)
        if slip < D_c:
            grad[0, 4] = -(mu_static - mu_dynamic) / D_c
        return grad

    stepper.mu_state_callback = mu_from_state
    stepper.dmu_dstate_callback = dmu_dstate
    y0 = np.zeros(5)

    y1, _aux1, info = stepper.step(
        0.0, y0, {"mu": np.array([mu_static])}, 0.01,
    )

    mu_expected = mu_from_state(y0 + stepper.theta * (y1 - y0))
    np.testing.assert_allclose(info["mu_used"], mu_expected, atol=1.0e-12)
    assert abs(info["mu_used"][0] - mu_static) > 1.0e-3
    assert dmu_calls["count"] > 0
    assert info["contact_ssn_converged"]


def test_descriptor_mjf_state_mu_uses_prestep_velocity_for_fremond_offset():
    # The Fremond restitution offset (theta(1+e)-1)*u_{N,k} must be built from
    # the PRE-STEP contact velocity D@y on the state-mu route, exactly as on
    # the aux-mu route -- not from the free predictor D@(z_pred - X@offset)
    # (the velocity with the whole contact reaction, including the prestress
    # hold, removed).  A critically prestressed contact in exact equilibrium
    # must STICK with zero motion; with the predictor datum it ruptures in one
    # step (the fictitious closing velocity turns into a restitution push that
    # the De Saxce coupling converts into slip).
    import scipy.sparse as sp

    mu0, N0 = 0.3, 1.0e6
    tau = 0.999 * mu0 * N0                  # prestress at 99.9% of the cone
    n = 2
    common = dict(
        A=sp.eye(n, format="csr"),
        rhs_callable=lambda t, y: np.zeros(n),
        rhs_jac_callable=lambda t, y: sp.csr_matrix((n, n)),
        D_extract=np.eye(n), B=np.eye(n),
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu0, "e": 0.0}],
        contact_offset_force=lambda y, t: np.array([N0, -tau]),
        theta=0.5, aux_law="constant", contact_solver="petsc_ssn",
        contact_linear_solver="dense", contact_residual="soc_fb",
        theta_linear_solver="scipy",
    )
    y0 = np.zeros(n)
    h = 1.0e-2

    ref = DescriptorMoreauJeanFremondStepper(**common)
    y_ref, _, info_ref = ref.step(0.0, y0, {"mu": np.array([mu0])}, h)

    st = DescriptorMoreauJeanFremondStepper(
        **common, mu_state_callback=lambda z, t: np.array([mu0]),
    )
    y_new, _, info = st.step(0.0, y0, {"mu": np.array([mu0])}, h)

    # equilibrium: stick, zero motion, reaction = prestress hold
    assert info["regime"] == ["stick"]
    np.testing.assert_allclose(y_new, np.zeros(n), atol=1.0e-9)
    # constant mu => the state-mu route must reproduce the aux-mu route
    np.testing.assert_allclose(y_new, y_ref, atol=1.0e-12)
    np.testing.assert_allclose(
        info["p_contact_effective"], info_ref["p_contact_effective"],
        rtol=1.0e-12,
    )


def test_solve_mjf_adaptive_sliding_block_matches_analytical():
    mass, g, mu = 1.0, 9.81, 0.4
    F_T = 1.5 * mu * mass * g
    stepper = _descriptor_sliding_block(mu, F_T)
    t_end = 0.05
    t, y, h, info, attempts = solve_mjf_adaptive(
        stepper, (0.0, t_end), np.zeros(4), {"mu": np.array([mu])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-9, h_max=t_end,
    )
    assert t[-1] == pytest.approx(t_end)
    np.testing.assert_allclose(np.diff(t), h, atol=1.0e-15)
    accel = (F_T - mu * mass * g) / mass
    assert y[-1, 3] == pytest.approx(accel * t_end, rel=2.0e-2)
    assert y[-1, 2] == pytest.approx(0.0, abs=1.0e-8)
    assert all("adaptive_error" in rec for rec in info)
    assert "p_contact_force" in info[-1]
    # Sliding contact sits on the cone edge with the mu used by the law.
    p_eff = info[-1]["p_contact_effective"]
    assert abs(abs(p_eff[1]) - mu * p_eff[0]) < 1.0e-8


def test_solve_mjf_adaptive_mu_fixed_point_slip_weakening():
    mass, g = 1.0, 9.81
    mu_s, mu_d, D_c = 0.6, 0.2, 1.0e-3
    F_T = 1.2 * mu_s * mass * g
    stepper = _descriptor_sliding_block(mu_s, F_T, n_slip=1)

    def mu_from_state(y_state):
        frac = min(float(y_state[4]) / D_c, 1.0)
        return np.array([mu_s - (mu_s - mu_d) * frac])

    t_end = 0.08
    t, y, h, info, attempts = solve_mjf_adaptive(
        stepper, (0.0, t_end), np.zeros(5), {"mu": np.array([mu_s])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-10, h_max=5.0e-3,
        mu_from_state=mu_from_state, mu_fixed_point_tol=1.0e-10,
    )
    assert t[-1] == pytest.approx(t_end)
    assert y[-1, 4] > D_c  # fully weakened
    assert info[-1]["mu_law"][0] == pytest.approx(mu_d)
    assert max(rec["mu_fixed_point_iters"] for rec in info) >= 1
    assert max(rec["mu_fixed_point_error"] for rec in info) <= 1.0e-9 + 1.0e-12
    # Reaction sits on the weakened cone: |p_t| = mu_law * p_n.
    p_eff = info[-1]["p_contact_effective"]
    mu_law = info[-1]["mu_law"][0]
    assert abs(abs(p_eff[1]) - mu_law * p_eff[0]) < 1.0e-8


def _feedback_block_stepper(*, lowrank):
    """Sliding block with a rank-1 feedback on the tangential momentum row.

    rhs[3] = F_T - k q_t - c v_t.  The feedback ``-(k, c)`` is rank-1:
    ``U = e_3``, ``V = [0, k, 0, c]``.  With ``lowrank=True`` rhs_jac omits it
    and it is passed via ``theta_lowrank_jac``; with ``lowrank=False`` rhs_jac
    folds it in (the dense reference).  Both must integrate identically.
    """
    mass, g, mu, F_T, k, c = 1.0, 9.81, 0.4, 6.0, 0.7, 0.3
    A = np.eye(4)

    def rhs(t, y):
        return np.array([y[2], y[3], -mass * g, F_T - k * y[1] - c * y[3]])

    if lowrank:
        def rhs_jac(t, y):
            J = np.zeros((4, 4)); J[0, 2] = 1.0; J[1, 3] = 1.0
            return J
        U = np.array([0.0, 0.0, 0.0, 1.0]); V = np.array([0.0, k, 0.0, c])
        extra = {"theta_lowrank_jac": (U, V)}
    else:
        def rhs_jac(t, y):
            J = np.zeros((4, 4)); J[0, 2] = 1.0; J[1, 3] = 1.0
            J[3, 1] = -k; J[3, 3] = -c
            return J
        extra = {}

    D = np.zeros((2, 4)); D[0, 2] = 1.0; D[1, 3] = 1.0
    B = np.zeros((4, 2)); B[2, 0] = 1.0; B[3, 1] = 1.0
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu, "e": 0.0}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy", **extra,
    )


def test_theta_lowrank_jac_matches_dense_feedback():
    # An explicit rank-1 theta_lowrank_jac update must reproduce, bit-for-bit,
    # the result of folding the same feedback into a dense Jacobian.
    st_lr = _feedback_block_stepper(lowrank=True)
    st_dn = _feedback_block_stepper(lowrank=False)
    y_lr = np.array([0.0, 0.0, -0.5, 1.0]); y_dn = y_lr.copy()
    aux_lr = {"mu": np.array([0.4])}; aux_dn = {"mu": np.array([0.4])}
    h = 0.01
    for kk in range(40):
        y_lr, aux_lr, _ = st_lr.step(kk * h, y_lr, aux_lr, h)
        y_dn, aux_dn, _ = st_dn.step(kk * h, y_dn, aux_dn, h)
        np.testing.assert_allclose(y_lr, y_dn, rtol=0.0, atol=1.0e-11)


def test_theta_lowrank_jac_requires_scipy_backend():
    with pytest.raises(ValueError, match="scipy"):
        st = _feedback_block_stepper(lowrank=True)
        st.theta_linear_solver = "iterative"
        st.step(0.0, np.array([0.0, 0.0, -0.5, 1.0]), {"mu": np.array([0.4])}, 0.01)


# ----------------------------------------------------------------------------
# Gap index set Ibar1_k (geometric activation) -- open/close contact.
# Acary & Collins-Craft 2025, Eq. 19: a contact enters the cone CP only when
# g(q_k) <= 0 and u_{N,k} <= 0.  Validated on a bouncing ball cast as a
# first-order descriptor, against the closed-form apex ratio e^2.
# ----------------------------------------------------------------------------
def _first_order_ball(e, *, gap=True, g=9.81):
    """Bouncing ball as a first-order descriptor: y=[q,v], A=I, rhs=[v,-g],
    normal contact u_N=v, gap g(q)=q.  gap=True turns on the Ibar1_k index set."""
    A = np.eye(2)

    def rhs(t, y):
        return np.array([y[1], -g])

    def rhs_jac(t, y):
        return np.array([[0.0, 1.0], [0.0, 0.0]])

    D = np.array([[0.0, 1.0]])      # u_N = v
    B = np.array([[0.0], [1.0]])    # reaction acts on the v equation
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu_init": 0.0, "e": e}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
        gap_callable=(lambda z: np.array([z[0]])) if gap else None,
    )


def _ball_trace(stepper, h, h0=1.0, t_end=1.4):
    y = np.array([h0, 0.0])
    aux = {"mu": np.array([0.0])}
    qs = [h0]
    for k in range(int(round(t_end / h))):
        y, aux, _ = stepper.step(k * h, y, aux, h)
        qs.append(y[0])
    qs = np.array(qs)
    imin = int(np.argmin(qs[: len(qs) // 2]))   # first-impact index
    return qs, imin


def test_descriptor_gap_index_set_bouncing_ball_converges():
    # Free flight (gap > 0 -> inactive), impact (gap <= 0 -> active), rebound to
    # e^2 h0.  Event-capturing: the apex error must shrink under h-refinement.
    g = 9.81; h0 = 1.0; e = 0.5
    t1 = np.sqrt(2 * h0 / g)
    errs = {}
    imin_fine = None
    for h in (4.0e-3, 1.0e-3):
        qs, imin = _ball_trace(_first_order_ball(e), h)
        errs[h] = abs(qs[imin:].max() - e ** 2 * h0)
        if h == 1.0e-3:
            imin_fine = imin
    assert errs[1.0e-3] < errs[4.0e-3]          # refinement converges
    assert errs[1.0e-3] < 0.01 * h0             # apex within 1% of e^2 h0
    assert abs(imin_fine * 1.0e-3 - t1) < 5.0e-3  # first impact near sqrt(2h0/g)


def test_descriptor_gap_index_set_inelastic_ball_rests():
    qs, imin = _ball_trace(_first_order_ball(0.0), 2.0e-3)
    assert qs[imin:].max() < 0.02              # inelastic: stays on the floor


def test_descriptor_without_gap_callable_is_persistent():
    # The SAME ball with gap_callable=None: the velocity-level contact is always
    # active (persistent contact), so the body is held at its start height.  This
    # is exactly the regime the gap index set generalises.
    qs, _ = _ball_trace(_first_order_ball(0.5, gap=False), 2.0e-3, t_end=0.6)
    assert np.all(np.abs(qs - 1.0) < 1.0e-2)


def test_descriptor_gap_persistent_contact_margin_stays_active():
    # A block resting on the floor under gravity is a PERSISTENT contact whose gap
    # sits at ~0 (only numerical noise).  A gap_tol margin above that noise keeps the
    # gap index set active, matching gap_callable=None exactly -- the analog of the
    # prestressed fault, where g(q_k) ~ 1e-10 and gap_tol=1e-6 holds it closed.
    g = 9.81
    A = np.eye(2)

    def rhs(t, y):
        return np.array([y[1], -g])

    def rhs_jac(t, y):
        return np.array([[0.0, 1.0], [0.0, 0.0]])

    D = np.array([[0.0, 1.0]]); B = np.array([[0.0], [1.0]])

    def build(gap, gtol):
        return DescriptorMoreauJeanFremondStepper(
            A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
            contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu_init": 0.0, "e": 0.0}],
            theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
            gap_callable=gap, gap_tol=gtol,
        )

    h = 0.01
    out = {}
    for tag, (gap, gtol) in {"none": (None, 0.0),
                             "gap": (lambda z: np.array([z[0]]), 1.0e-6)}.items():
        st = build(gap, gtol)
        y = np.array([0.0, 0.0]); aux = {"mu": np.array([0.0])}
        for k in range(100):
            y, aux, _ = st.step(k * h, y, aux, h)
        out[tag] = y.copy()
    # the gap+margin path is byte-identical to persistent (all-active) and rests
    np.testing.assert_allclose(out["none"], out["gap"], atol=1e-12)
    assert abs(out["none"][0]) < 1e-6 and abs(out["none"][1]) < 1e-6


# ----------------------------------------------------------------------------
# Combined projection (GGL position admissibility around the Fremond velocity
# law).  After the velocity solve, project q_{k+1} so g(q) >= 0 via a separate
# projection multiplier nu -- never a contact impulse, never a rebound velocity.
# This removes gap drift/penetration WITHOUT the spurious kinetic energy that the
# naive u_N + g/h injects (which makes an inelastic ball bounce).
# ----------------------------------------------------------------------------
def _proj_ball(e, *, projection, g=9.81, max_iter=3):
    A = np.eye(2)

    def rhs(t, y):
        return np.array([y[1], -g])

    def rhs_jac(t, y):
        return np.array([[0.0, 1.0], [0.0, 0.0]])

    D = np.array([[0.0, 1.0]]); B = np.array([[0.0], [1.0]])
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu_init": 0.0, "e": e}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
        gap_callable=lambda z: np.array([z[0]]),
        gap_jac=(np.array([[1.0, 0.0]]) if projection else None),
        combined_projection=projection,
        projection_tol=1.0e-12,
        combined_projection_max_iter=max_iter,
    )


def _proj_trace(stepper, h, h0=1.0, t_end=1.4):
    y = np.array([h0, 0.0]); aux = {"mu": np.array([0.0])}
    qs = [h0]; pen_before = []
    for k in range(int(round(t_end / h))):
        y, aux, info = stepper.step(k * h, y, aux, h)
        qs.append(y[0])
        if "proj_penetration_before" in info:
            pen_before.append(info["proj_penetration_before"])
    return np.array(qs), np.array(pen_before)


def test_combined_projection_removes_penetration_drift():
    # The velocity-only scheme lets the ball sink O(h) below the floor each impact
    # (Moreau penetration); combined projection drives the gap back to ~0 every step
    # while the elastic apex stays at e^2 h0.
    h = 2.0e-3; e = 0.5
    q_v, _ = _proj_trace(_proj_ball(e, projection=False), h)
    q_p, pen = _proj_trace(_proj_ball(e, projection=True), h)
    assert q_v.min() < -1.0e-3                  # velocity-only penetrates
    assert q_p.min() > -1.0e-9                  # projected: penetration removed
    assert pen.size and np.abs(pen).max() > 1.0e-4   # projection actually fired
    imin = int(np.argmin(q_p[: len(q_p) // 2]))
    assert abs(q_p[imin:].max() - e ** 2 * 1.0) < 0.01   # apex still e^2 h0


def test_combined_projection_inelastic_ball_rests_no_rebound():
    # The case the naive g/h correction fails: an inelastic (e=0) ball must REST.
    # Because the projection multiplier is kept out of the velocity, no rebound
    # energy is injected -> the ball stays on the floor (unlike g/h, which bounces).
    h = 2.0e-3
    q_p, pen = _proj_trace(_proj_ball(0.0, projection=True), h)
    imin = int(np.argmin(q_p[: len(q_p) // 2]))
    assert q_p.min() > -1.0e-9                   # no penetration
    assert q_p[imin:].max() < 1.0e-6             # rests on the floor (no bounce)
    assert np.abs(pen).max() > 1.0e-4            # projection was correcting drift


def test_combined_projection_off_matches_no_projection():
    # combined_projection=False must reproduce the plain gap-index-set path exactly,
    # even with gap_jac supplied (it is simply unused).
    h = 2.0e-3
    A = np.eye(2)
    rhs = lambda t, y: np.array([y[1], -9.81])
    rhs_jac = lambda t, y: np.array([[0.0, 1.0], [0.0, 0.0]])
    D = np.array([[0.0, 1.0]]); B = np.array([[0.0], [1.0]])

    def build(proj_kwargs):
        return DescriptorMoreauJeanFremondStepper(
            A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
            contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu_init": 0.0, "e": 0.5}],
            theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
            gap_callable=lambda z: np.array([z[0]]), **proj_kwargs,
        )

    out = {}
    for tag, kw in {"plain": {},
                    "off": dict(gap_jac=np.array([[1.0, 0.0]]), combined_projection=False)}.items():
        st = build(kw); y = np.array([1.0, 0.0]); aux = {"mu": np.array([0.0])}
        for k in range(700):
            y, aux, _ = st.step(k * h, y, aux, h)
        out[tag] = y.copy()
    np.testing.assert_allclose(out["plain"], out["off"], rtol=0.0, atol=0.0)


def test_combined_projection_requires_gap_jac():
    with pytest.raises(ValueError, match="gap_jac"):
        DescriptorMoreauJeanFremondStepper(
            A=np.eye(2), rhs_callable=lambda t, y: y, rhs_jac_callable=lambda t, y: np.eye(2),
            D_extract=np.array([[0.0, 1.0]]), B=np.array([[0.0], [1.0]]),
            contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [], "mu_init": 0.0, "e": 0.0}],
            gap_callable=lambda z: np.array([z[0]]), combined_projection=True,
        )


# ----------------------------------------------------------------------------
# Total-energy verification on a 2-D frictional bouncing ball (normal+tangential).
# The Fremond scheme must satisfy the discrete energy balance exactly at theta=1/2:
#   E_{k+1} - E_k == W_contact = u_{k+theta} . p_contact   (the average velocity
# conjugate to the impulse), with W_contact <= 0 (dissipative) and E never rising.
# Probed for restitution e in {0, 1/2, 1}, frictionless and frictional (take-off
# vs impact-with-friction).
# ----------------------------------------------------------------------------
G_BALL = 9.81


def _frictional_ball(e, mu, *, projection=False):
    # First-order descriptor: state [q_n, v_n, v_t], A=I, rhs=[v_n, -g, 0];
    # normal velocity u_N=v_n, tangential u_T=v_t; gap g(q)=q_n.
    A = np.eye(3)

    def rhs(t, y):
        return np.array([y[1], -G_BALL, 0.0])

    def rhs_jac(t, y):
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    D = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])     # u_N=v_n, u_T=v_t
    B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])   # p_N->v_n, p_T->v_t
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu_init": mu, "e": e}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
        contact_residual="soc_fb",
        gap_callable=lambda z: np.array([z[0]]),
        gap_jac=(np.array([[1.0, 0.0, 0.0]]) if projection else None),
        combined_projection=projection,
    )


def _ball_energy_audit(st, h=1.0e-3, h0=1.0, vt0=3.0, t_end=1.2):
    Etot = lambda y: 0.5 * (y[1] ** 2 + y[2] ** 2) + G_BALL * y[0]
    y = np.array([h0, 0.0, vt0]); aux = {"mu": np.array([st.mu_init[0]])}
    E0 = Etot(y); max_dE = -np.inf; max_W = -np.inf; max_ident = 0.0
    for k in range(int(round(t_end / h))):
        Ek = Etot(y)
        y, aux, info = st.step(k * h, y, aux, h)
        u = info["u_kt"]; p = info["p_contact"]
        W = float(u @ p) if u.size else 0.0
        max_dE = max(max_dE, Etot(y) - Ek)
        max_W = max(max_W, W)
        max_ident = max(max_ident, abs((Etot(y) - Ek) - W))
        y_end = y
    return dict(E0=E0, Efin=Etot(y_end), dissip=E0 - Etot(y_end),
                max_dE=max_dE, max_W=max_W, max_ident=max_ident, vt=y_end[2])


def test_energy_frictional_ball_balance_is_exact_and_dissipative():
    # Pure Fremond scheme: energy balance exact, contact work <=0, E non-increasing.
    for e in (0.0, 0.5, 1.0):
        for mu in (0.0, 0.3):
            r = _ball_energy_audit(_frictional_ball(e, mu))
            assert r["max_ident"] < 1e-10, (e, mu, r)   # E_{k+1}-E_k == u_kt.p exactly
            assert r["max_W"] < 1e-10, (e, mu, r)       # contact work non-positive
            assert r["max_dE"] < 1e-10, (e, mu, r)      # total energy never increases


def test_energy_frictional_ball_physics():
    # e=1 frictionless take-off: lossless -> energy conserved, slide preserved.
    r = _ball_energy_audit(_frictional_ball(1.0, 0.0))
    assert abs(r["dissip"]) < 1e-6
    assert abs(r["vt"] - 3.0) < 1e-6
    # e=1 impact WITH friction: normal lossless yet friction dissipates the slide.
    r = _ball_energy_audit(_frictional_ball(1.0, 0.3))
    assert r["dissip"] > 1.0
    assert abs(r["vt"]) < 3.0
    # frictionless: more restitution -> less dissipation.
    d = [_ball_energy_audit(_frictional_ball(e, 0.0))["dissip"] for e in (0.0, 0.5, 1.0)]
    assert d[0] > d[1] > d[2] - 1e-9
    assert abs(d[2]) < 1e-6


def test_energy_combined_projection_keeps_impulse_dissipative():
    # With combined projection ON the Fremond impulse work stays <=0; the only
    # energy added is the bounded one-time position-projection lift (potential).
    for e in (0.0, 0.5, 1.0):
        r = _ball_energy_audit(_frictional_ball(e, 0.3, projection=True))
        assert r["max_W"] < 1e-10            # impulse still dissipative
        assert r["max_dE"] < 0.05            # any energy gain bounded by the lift


def test_combined_projection_does_not_perturb_velocity_solution():
    # Consistency guarantee: position projection only modifies the END-OF-STEP
    # position handed to the next step.  Within a step it must leave the Fremond
    # velocity solution (p_contact, u_{k+theta}) byte-identical and never touch a
    # velocity DOF -> it cannot break the SOCCP this step.  Step from the SAME
    # penetrating IC with projection on vs off and compare.
    y0 = np.array([-1.0e-3, -4.4, 1.0])     # penetrated, descending, sliding
    aux = {"mu": np.array([0.3])}
    st_off = _frictional_ball(0.5, 0.3, projection=False)
    st_on = _frictional_ball(0.5, 0.3, projection=True)
    y_off, _, i_off = st_off.step(0.0, y0.copy(), dict(aux), 1.0e-3)
    y_on, _, i_on = st_on.step(0.0, y0.copy(), dict(aux), 1.0e-3)
    # identical velocity-level solution
    np.testing.assert_allclose(i_on["p_contact"], i_off["p_contact"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(i_on["u_kt"], i_off["u_kt"], rtol=0.0, atol=0.0)
    # only the position DOF differs; velocity DOFs are untouched by the projection
    np.testing.assert_allclose(y_on[1:], y_off[1:], rtol=0.0, atol=0.0)
    assert y_on[0] >= -1.0e-12               # endpoint gap admissible after projection
    assert i_on["proj_penetration_after"] >= -1.0e-12
