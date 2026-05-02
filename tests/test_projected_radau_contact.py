import numpy as np
from scipy.sparse import csr_matrix

from solve_nivp.integrations import RadauIIA
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projected_radau_contact import (
    DeSaxceProjectedConeLaw,
    build_projected_radau_contact,
)
from solve_nivp.projections import IdentityProjection


def _zero_rhs(t, y, *extra):
    return np.zeros_like(y, dtype=float)


def _zero_jac(t, y, *extra):
    return np.zeros((len(y), len(y)), dtype=float)


def test_projected_radau_constraints_patch_stage_rhs_and_jacobian():
    def rhs(t, y, *extra):
        return np.array([5.0, 99.0])

    def jac(t, y, *extra):
        return np.array([[7.0, 8.0], [9.0, 10.0]])

    def g(y):
        return np.array([2.0 * y[0] + 1.0])

    def dg(y):
        return np.array([[2.0]])

    cs = build_projected_radau_contact(
        np.eye(2),
        rhs,
        np.array([0.0, 0.0]),
        [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=np.eye(2),
        D_extract=np.eye(2),
        B=np.array([[2.0], [3.0]]),
        constraints=[
            {"g": g, "dg_dy": dg, "y_slice": slice(0, 1), "q_slice": slice(1, 2)}
        ],
        rhs_jac=jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact
    y = np.array([3.0, 10.0])

    np.testing.assert_allclose(
        model._smooth_rhs(0.0, y, h=0.25, prev_state=np.zeros(2)),
        [5.0, -3.0],
    )
    np.testing.assert_allclose(
        model._smooth_jac(0.0, y, h=0.25, prev_state=np.zeros(2)).toarray(),
        [[7.0, 8.0], [2.0, -1.0]],
    )
    B_coupling = (
        model._B_coupling.toarray()
        if hasattr(model._B_coupling, "toarray")
        else np.asarray(model._B_coupling)
    )
    np.testing.assert_allclose(B_coupling, [[2.0], [0.0]])

    rk_A = np.array([[5.0 / 12.0, -1.0 / 12.0], [3.0 / 4.0, 1.0 / 4.0]])
    rk_b = np.array([3.0 / 4.0, 1.0 / 4.0])
    rk_c = np.array([1.0 / 3.0, 1.0])
    Z = model.pack([y, y], [np.array([0.4]), np.array([0.7])])
    F = model.residual(Z, 0.0, np.zeros(2), 1.0, rk_A, rk_b, rk_c)
    np.testing.assert_allclose(F[1], 10.0 + 3.0 * rk_c[0])
    np.testing.assert_allclose(F[3], 10.0 + 3.0 * rk_c[1])

    J = model.jacobian(Z, 0.0, np.zeros(2), 1.0, rk_A, rk_b, rk_c).toarray()
    np.testing.assert_allclose(J[[1, 3], 4:6], 0.0)


def test_auto_scaling_uses_actual_traction_coupling():
    cs = build_projected_radau_contact(
        np.diag([2.0, 4.0]),
        _zero_rhs,
        np.array([0.0, 0.0]),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.5}],
        C_extract=np.eye(2),
        D_extract=np.eye(2),
        B=np.array([[10.0, 0.0], [0.0, 0.25]]),
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r="auto",
        friction_r="auto",
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact

    np.testing.assert_allclose(model._auto_normal_r_base, [5.0])
    np.testing.assert_allclose(model._auto_friction_r_base, [0.0625])


def test_desaxce_projected_law_matches_finite_difference_jacobian():
    law = DeSaxceProjectedConeLaw()
    normal_quantity = 0.1
    velocity = np.array([99.0, 0.2])
    percussion = np.array([0.4, 0.8])
    mu = 0.5
    normal_scale = 1.0
    friction_scale = 1.0

    f, df_dnormal, df_du, df_dr = law.residual_and_jac(
        normal_quantity,
        velocity,
        percussion,
        mu,
        normal_scale,
        friction_scale,
    )
    assert f.shape == (2,)

    def eval_block(nq, vel, pi):
        return law.residual_and_jac(
            nq,
            vel,
            pi,
            mu,
            normal_scale,
            friction_scale,
        )[0]

    eps = 1.0e-6
    fd_n = (
        eval_block(normal_quantity + eps, velocity, percussion)
        - eval_block(normal_quantity - eps, velocity, percussion)
    ) / (2.0 * eps)
    np.testing.assert_allclose(df_dnormal, fd_n, rtol=1.0e-6, atol=1.0e-8)

    fd_u = np.zeros_like(df_du)
    for j in range(velocity.size):
        v_plus = velocity.copy()
        v_minus = velocity.copy()
        v_plus[j] += eps
        v_minus[j] -= eps
        fd_u[:, j] = (
            eval_block(normal_quantity, v_plus, percussion)
            - eval_block(normal_quantity, v_minus, percussion)
        ) / (2.0 * eps)
    np.testing.assert_allclose(df_du, fd_u, rtol=1.0e-6, atol=1.0e-8)

    fd_r = np.zeros_like(df_dr)
    for j in range(percussion.size):
        p_plus = percussion.copy()
        p_minus = percussion.copy()
        p_plus[j] += eps
        p_minus[j] -= eps
        fd_r[:, j] = (
            eval_block(normal_quantity, velocity, p_plus)
            - eval_block(normal_quantity, velocity, p_minus)
        ) / (2.0 * eps)
    np.testing.assert_allclose(df_dr, fd_r, rtol=1.0e-6, atol=1.0e-8)


def test_desaxce_projected_law_self_dual_sliding_zero():
    law = DeSaxceProjectedConeLaw(rho=0.7)

    f, *_ = law.residual_and_jac(
        normal_quantity=0.0,
        contact_velocity=np.array([0.0, 2.0]),
        percussion=np.array([4.0, -1.0]),
        mu=0.25,
        normal_scale=1.0,
        friction_scale=1.0,
    )

    np.testing.assert_allclose(f, [0.0, 0.0], atol=1.0e-14)


def test_desaxce_projected_law_keeps_separated_contact_inactive_with_unequal_scales():
    law = DeSaxceProjectedConeLaw()

    f, df_dnormal, df_du, df_dr = law.residual_and_jac(
        normal_quantity=1.0,
        contact_velocity=np.array([0.0, 0.4]),
        percussion=np.zeros(2),
        mu=0.5,
        normal_scale=0.1,
        friction_scale=10.0,
    )

    np.testing.assert_allclose(f, [0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(df_dnormal, [0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(df_du, np.zeros((2, 2)), atol=1.0e-14)
    np.testing.assert_allclose(df_dr, np.eye(2), atol=1.0e-14)


def test_projected_radau_contact_jacobian_includes_dmu_dy():
    def mu(y):
        return 0.5 + 0.1 * y[2]

    def dmu_dy(y):
        out = np.zeros_like(y, dtype=float)
        out[2] = 0.1
        return out

    C = np.array([[1.0, 0.0, 0.0]])
    D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cs = build_projected_radau_contact(
        np.eye(3),
        _zero_rhs,
        np.zeros(3),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": mu, "dmu_dy": dmu_dy}],
        C_extract=C,
        D_extract=D,
        B=D.T,
        rhs_jac=_zero_jac,
        contact_law="desaxce",
        desaxce_rho=0.7,
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact
    y = np.array([0.0, 0.2, 0.4])
    percussion = np.array([0.1, 0.0])

    Jy, _Jr = model._contact_jacobian(
        y, 0.0, percussion, np.zeros(2), 1.0, endpoint=False
    )
    eps = 1.0e-6
    y_plus = y.copy()
    y_minus = y.copy()
    y_plus[2] += eps
    y_minus[2] -= eps
    fd = (
        model._contact_residual(y_plus, 0.0, percussion, np.zeros(2), 1.0, endpoint=False)
        - model._contact_residual(y_minus, 0.0, percussion, np.zeros(2), 1.0, endpoint=False)
    ) / (2.0 * eps)

    np.testing.assert_allclose(Jy.toarray()[:, 2], fd, rtol=1.0e-6, atol=1.0e-8)
    assert abs(Jy.toarray()[1, 2]) > 1.0e-4


def test_projected_radau_desaxce_string_dispatch():
    cs = build_projected_radau_contact(
        np.eye(2),
        _zero_rhs,
        np.array([0.0, 0.0]),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.5}],
        C_extract=np.eye(2),
        D_extract=np.eye(2),
        B=np.eye(2),
        rhs_jac=_zero_jac,
        contact_law="de_saxce",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )

    assert isinstance(cs.projected_radau_contact.contact_law, DeSaxceProjectedConeLaw)


def test_desaxce_endpoint_projection_handles_frictionless_contact():
    cs = build_projected_radau_contact(
        np.eye(2),
        _zero_rhs,
        np.array([-1.0, 2.0]),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.0}],
        C_extract=np.eye(2),
        D_extract=np.eye(2),
        B=np.eye(2),
        rhs_jac=_zero_jac,
        contact_law="desaxce",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact

    y_plus, delta_pi, total_pi, total_eff, reported, ok, err = model.project_endpoint(
        np.array([-1.0, 2.0]),
        np.array([-1.0, 2.0]),
        np.zeros(2),
        0.1,
        0.1,
    )

    assert ok, err
    np.testing.assert_allclose(y_plus, [0.0, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(delta_pi, [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(total_pi, [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(total_eff, [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(reported, [1.0, 0.0], atol=1.0e-12)


def test_stage_contact_law_uses_accumulated_butcher_percussion():
    cs = build_projected_radau_contact(
        np.eye(1),
        _zero_rhs,
        np.array([1.0]),
        [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=np.eye(1),
        D_extract=np.eye(1),
        B=np.ones((1, 1)),
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact
    rk_A = np.array([[5.0 / 12.0, -1.0 / 12.0], [3.0 / 4.0, 1.0 / 4.0]])
    rk_b = np.array([3.0 / 4.0, 1.0 / 4.0])
    rk_c = np.array([1.0 / 3.0, 1.0])

    y_stage = [np.array([1.0]), np.array([1.0])]
    dpi_base = [np.array([0.3]), np.array([0.0])]
    dpi_changed = [np.array([0.3]), np.array([0.12])]

    F_base = model.residual(
        model.pack(y_stage, dpi_base),
        0.0,
        np.array([1.0]),
        1.0,
        rk_A,
        rk_b,
        rk_c,
    )
    F_changed = model.residual(
        model.pack(y_stage, dpi_changed),
        0.0,
        np.array([1.0]),
        1.0,
        rk_A,
        rk_b,
        rk_c,
    )

    stage_1_contact_row = 2 * model.n_phys
    expected = rk_A[0, 1] * 0.12
    np.testing.assert_allclose(
        F_changed[stage_1_contact_row] - F_base[stage_1_contact_row],
        expected,
        atol=1.0e-14,
    )


def test_endpoint_projection_is_inelastic_by_default():
    cs = build_projected_radau_contact(
        np.eye(1),
        _zero_rhs,
        np.array([-1.0]),
        [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=np.eye(1),
        D_extract=np.eye(1),
        B=np.ones((1, 1)),
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact

    y_plus, delta_pi, total_pi, total_eff, reported, ok, err = model.project_endpoint(
        np.array([-1.0]),
        np.array([-1.0]),
        np.zeros(1),
        0.1,
        0.1,
    )

    assert ok, err
    np.testing.assert_allclose(y_plus, [0.0], atol=1.0e-12)
    np.testing.assert_allclose(delta_pi, [1.0], atol=1.0e-12)
    np.testing.assert_allclose(total_pi, [1.0], atol=1.0e-12)
    np.testing.assert_allclose(total_eff, [1.0], atol=1.0e-12)
    np.testing.assert_allclose(reported, [1.0], atol=1.0e-12)


def test_projection_map_uses_full_mass_inverse_by_default():
    A = np.array(
        [
            [2.0, 0.5, 0.0],
            [0.5, 3.0, 0.25],
            [0.0, 0.25, 4.0],
        ]
    )
    B = np.array([[1.0], [0.0], [0.0]])
    cs = build_projected_radau_contact(
        A,
        _zero_rhs,
        np.zeros(3),
        [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=np.eye(3),
        D_extract=np.eye(3),
        B=B,
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )

    idx, P = cs.projected_radau_contact._projection_map()

    np.testing.assert_array_equal(idx, np.arange(3))
    np.testing.assert_allclose(P, np.linalg.solve(A, B), atol=1.0e-14)
    assert not np.isclose(P[0, 0], B[0, 0] / A[0, 0])


def test_projection_map_reduces_sparse_descriptor_matrix_by_default():
    A = csr_matrix(np.diag([2.0, 0.0, 3.0, 0.0]))
    B = np.array(
        [
            [2.0, 0.0],
            [7.0, 11.0],
            [0.0, 3.0],
            [5.0, 13.0],
        ]
    )
    cs = build_projected_radau_contact(
        A,
        _zero_rhs,
        np.zeros(4),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.5}],
        C_extract=np.eye(4),
        D_extract=np.eye(4),
        B=B,
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )

    idx, P = cs.projected_radau_contact._projection_map()

    np.testing.assert_array_equal(idx, [0, 2])
    expected = np.zeros((4, 2))
    expected[np.ix_([0, 2], [0, 1])] = np.linalg.solve(
        np.diag([2.0, 3.0]),
        B[[0, 2], :],
    )
    np.testing.assert_allclose(P, expected, atol=1.0e-14)


def test_endpoint_friction_impulse_satisfies_cone_and_opposes_slip():
    cs = build_projected_radau_contact(
        np.eye(2),
        _zero_rhs,
        np.array([-1.0, 2.0]),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.5}],
        C_extract=np.eye(2),
        D_extract=np.eye(2),
        B=np.eye(2),
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
    )
    model = cs.projected_radau_contact

    y_plus, _delta_pi, _total_pi, total_eff, _reported, ok, err = model.project_endpoint(
        np.array([-1.0, 2.0]),
        np.array([-1.0, 2.0]),
        np.zeros(2),
        0.1,
        0.1,
    )

    assert ok, err
    assert total_eff[0] >= -1.0e-12
    assert abs(total_eff[1]) <= 0.5 * total_eff[0] + 1.0e-10
    assert total_eff[1] * y_plus[1] <= 1.0e-12


def test_projected_radau_dispatch_handles_sparse_descriptor_endpoint_projection():
    def rhs(t, y, *extra):
        return np.array([0.0, y[1] - 7.0])

    def jac(t, y, *extra):
        return np.array([[0.0, 0.0], [0.0, 1.0]])

    cs = build_projected_radau_contact(
        csr_matrix(np.diag([1.0, 0.0])),
        rhs,
        np.array([-1.0, 7.0]),
        [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=csr_matrix([[1.0, 0.0]]),
        D_extract=csr_matrix([[1.0, 0.0]]),
        B=csr_matrix([[1.0], [0.0]]),
        rhs_jac=jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="force",
    )
    solver = ImplicitEquationSolver(
        method="semismooth_newton",
        proj=cs.projection,
        component_slices=cs.component_slices,
        tol=1.0e-12,
    )
    solver.rhs_jacobian = cs.rhs_jac
    projected = RadauIIA(solver=solver, A=cs.A, **cs.integrator_opts)

    y_new, Fk_new, err, ok, _iters = projected.step(cs.rhs, 0.0, cs.y0.copy(), 0.1)

    assert ok
    np.testing.assert_allclose(y_new[:2], [0.0, 7.0], atol=1.0e-10)
    np.testing.assert_allclose(cs.projected_radau_contact.last_total_pi, [1.0], atol=1.0e-12)
    assert Fk_new.shape == y_new.shape
    assert err.shape == y_new.shape


def test_endpoint_natural_map_does_not_hard_open_on_positive_roundoff_gap():
    C = np.array([[0.0, 1.0]])
    D = np.array([[1.0, 0.0]])
    common = dict(
        A=np.eye(2),
        rhs_smooth=_zero_rhs,
        y0=np.array([0.0, 0.0]),
        contacts=[{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=C,
        D_extract=D,
        B=D.T,
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="impulse",
        gap_tol=0.0,
    )
    y_plus = np.array([0.0, 1.0e-16])
    total_eff = np.array([100.0])

    hard_gate = build_projected_radau_contact(
        **common,
        endpoint_inactive_handling="gap",
    ).projected_radau_contact
    cone_map = build_projected_radau_contact(
        **common,
        endpoint_inactive_handling="natural_map",
    ).projected_radau_contact

    np.testing.assert_allclose(
        hard_gate._endpoint_contact_residual(
            y_plus, np.zeros(2), total_eff, 0.0, 1.0
        ),
        [100.0],
    )
    np.testing.assert_allclose(
        cone_map._endpoint_contact_residual(
            y_plus, np.zeros(2), total_eff, 0.0, 1.0
        ),
        [0.0],
        atol=1.0e-14,
    )


def test_projected_radau_dispatch_runs_without_changing_plain_radau():
    cs = build_projected_radau_contact(
        np.eye(1),
        _zero_rhs,
        np.array([-1.0]),
        [{"vel_normal_idx": 0, "mu": 0.0}],
        C_extract=np.eye(1),
        D_extract=np.eye(1),
        B=np.ones((1, 1)),
        rhs_jac=_zero_jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="force",
    )
    solver = ImplicitEquationSolver(
        method="semismooth_newton",
        proj=cs.projection,
        component_slices=cs.component_slices,
        tol=1.0e-12,
    )
    solver.rhs_jacobian = cs.rhs_jac
    projected = RadauIIA(solver=solver, A=cs.A, **cs.integrator_opts)

    y_new, Fk_new, err, ok, _iters = projected.step(cs.rhs, 0.0, cs.y0.copy(), 0.1)

    assert ok
    np.testing.assert_allclose(y_new[:1], [0.0], atol=1.0e-10)
    assert Fk_new.shape == y_new.shape
    assert err.shape == y_new.shape
    assert projected.projected_radau_contact is cs.projected_radau_contact

    plain_solver = ImplicitEquationSolver(
        method="semismooth_newton",
        proj=IdentityProjection(),
        tol=1.0e-12,
    )
    plain = RadauIIA(stages=2, solver=plain_solver)
    assert plain.projected_radau_contact is None


def test_bouncing_ball_projected_radau_is_inelastic_after_impact():
    mass = 1.0
    gravity = 9.81
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y, *extra):
        return np.array([0.0, -mass * gravity, y[0], y[1]])

    def jac(t, y, *extra):
        J = np.zeros((4, 4))
        J[2, 0] = 1.0
        J[3, 1] = 1.0
        return J

    C = np.zeros((2, 4))
    C[0, 3] = 1.0
    C[1, 2] = 1.0
    D = np.zeros((2, 4))
    D[0, 1] = 1.0
    D[1, 0] = 1.0

    cs = build_projected_radau_contact(
        A,
        rhs,
        np.array([0.0, 0.0, 0.0, 0.5]),
        [{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.0}],
        C_extract=csr_matrix(C),
        D_extract=csr_matrix(D),
        B=csr_matrix(D.T),
        rhs_jac=jac,
        contact_law="minimum_map",
        normal_r=1.0,
        friction_r=1.0,
        reported_reaction_units="force",
    )
    solver = ImplicitEquationSolver(
        method="semismooth_newton",
        proj=cs.projection,
        component_slices=cs.component_slices,
        tol=1.0e-11,
        max_iter=80,
        linear_solver="splu",
    )
    solver.rhs_jacobian = cs.rhs_jac
    integrator = RadauIIA(solver=solver, A=cs.A, **cs.integrator_opts)

    y = cs.y0.copy()
    t = 0.0
    h = 0.01
    y_hist = []
    for _ in range(60):
        y, _Fk, _err, ok, _iters = integrator.step(cs.rhs, t, y, h)
        assert ok
        t += h
        y_hist.append(y.copy())

    arr = np.asarray(y_hist)
    impact_idx = int(np.argmax(arr[:, 3] <= 1.0e-12))
    assert arr[impact_idx, 3] <= 1.0e-12
    assert arr[:, 3].min() >= -1.0e-10
    assert np.max(np.abs(arr[impact_idx:, 1])) <= 1.0e-10
    np.testing.assert_allclose(arr[-1, 4:], [mass * gravity, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(
        cs.projected_radau_contact.last_total_pi,
        [mass * gravity * h, 0.0],
        atol=1.0e-12,
    )
