import sys
import inspect
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

pytestmark = (pytest.mark.examples, pytest.mark.external)

helper_mod = pytest.importorskip(
    "sliding_block_one_step_patch_test_alart_curnier",
    reason="requires the prestressed fault example helper module",
)
_trim_failed_fixed_step_history = helper_mod._trim_failed_fixed_step_history
build_demo_contact_system = helper_mod.build_demo_contact_system
build_prestressed_fault_dynamic_context = helper_mod.build_prestressed_fault_dynamic_context
make_fault_mu_patch_callback = helper_mod.make_fault_mu_patch_callback
make_fault_prestress_patch_callbacks = helper_mod.make_fault_prestress_patch_callbacks
make_minimal_dynamic_fault_bc = helper_mod.make_minimal_dynamic_fault_bc
run_one_step_case = helper_mod.run_one_step_case
run_prestressed_fault_dynamic_history_case = helper_mod.run_prestressed_fault_dynamic_history_case
run_time_history_case = helper_mod.run_time_history_case


def test_fault_prestress_patch_callbacks_force_one_patch_contact():
    s = np.array([0.0, 0.2, 0.4, 0.6], dtype=float)
    out = make_fault_prestress_patch_callbacks(
        s,
        normal_prestress=1.0e-5,
        mu_friction=0.6,
        background_ratio=0.95,
        patch_ratio=1.01,
        patch_center=0.31,
        patch_half_width=1.0e-4,
    )
    mask = out["patch_mask"]
    assert mask.sum() == 1
    assert int(np.argmin(np.abs(s - 0.31))) == int(np.flatnonzero(mask)[0])
    np.testing.assert_allclose(
        out["tau_profile"][mask][0],
        1.01 * 0.6 * 1.0e-5,
    )


def test_build_prestressed_fault_dynamic_context_smoke():
    ctx = build_prestressed_fault_dynamic_context(
        n_elem=8,
        normal_prestress=1.0e-6,
        patch_ratio=0.99,
    )
    assert ctx["dynamic"] is True
    assert ctx["reaction_units"] == "force"
    assert abs(ctx["contact_plot"]["crack_angle_deg"]) < 1.0e-8
    assert np.count_nonzero(ctx["prestress_profile"]["patch_mask"]) >= 1
    assert ctx["velocity_slice"] is not None
    assert ctx["contact_velocity_extract"] is not None
    assert ctx["preload_equilibrium"] is False
    assert ctx["contact_backend_opts"]["offset_coupling_mode"] == "incremental_reference"
    assert "get_s0_load" in ctx["contact_backend_opts"]
    assert "get_w0_load" in ctx["contact_backend_opts"]
    assert "get_s0_ref" in ctx["contact_backend_opts"]
    assert "get_w0_ref" in ctx["contact_backend_opts"]
    assert ctx["preload_total_reactions"] is None
    np.testing.assert_allclose(ctx["cs"].y0[ctx["n_orig"] :], 0.0)


def test_fault_mu_patch_callback_force_one_patch_contact():
    s = np.array([0.0, 0.2, 0.4, 0.6], dtype=float)
    out = make_fault_mu_patch_callback(
        s,
        mu_friction=0.6,
        patch_mu_ratio=0.9,
        patch_center=0.31,
        patch_half_width=1.0e-4,
    )
    mask = out["patch_mask"]
    assert mask.sum() == 1
    assert int(np.argmin(np.abs(s - 0.31))) == int(np.flatnonzero(mask)[0])
    np.testing.assert_allclose(out["mu_profile"][mask][0], 0.54)


def test_build_prestressed_fault_dynamic_context_weakening_mode_sets_mu_patch():
    ctx = build_prestressed_fault_dynamic_context(
        n_elem=8,
        normal_prestress=1.0e-6,
        background_ratio=0.95,
        nucleation_mode="weakening",
        patch_mu_ratio=0.9,
    )
    assert ctx["nucleation_mode"] == "weakening"
    assert "mu_patch_profile" in ctx
    mu_profile = ctx["mu_patch_profile"]["mu_profile"]
    assert np.min(mu_profile) < np.max(mu_profile)
    assert ctx["reaction_nl_atol"] <= 1.0e-10


def test_build_demo_contact_system_accepts_per_contact_mu_array():
    preview = build_demo_contact_system(
        mu_friction=0.6,
        initial_gap_phys=0.0,
        reverse_gap_sign=False,
        rho_g=0.0,
        n_elem=8,
        element_type="tri",
        crack_theta=np.pi / 2.0,
        crack_x0=0.0,
        crack_y0=0.0,
        crack_length=0.6,
        bc_full=make_minimal_dynamic_fault_bc(),
        dynamic=True,
    )
    mu_vals = np.linspace(0.5, 0.7, int(preview["n_c"]))
    with np.errstate(all="ignore"):
        ctx = build_demo_contact_system(
            mu_friction=mu_vals,
            initial_gap_phys=0.0,
            reverse_gap_sign=False,
            rho_g=0.0,
            n_elem=8,
            element_type="tri",
            crack_theta=np.pi / 2.0,
            crack_x0=0.0,
            crack_y0=0.0,
            crack_length=0.6,
            bc_full=make_minimal_dynamic_fault_bc(),
            dynamic=True,
        )
    assert ctx["dynamic"] is True


def test_build_demo_contact_system_auto_tightens_dynamic_reaction_tolerance():
    preview = build_demo_contact_system(
        mu_friction=0.6,
        initial_gap_phys=0.0,
        reverse_gap_sign=False,
        rho_g=0.0,
        n_elem=8,
        element_type="tri",
        crack_theta=np.pi / 2.0,
        crack_x0=0.0,
        crack_y0=0.0,
        crack_length=0.6,
        bc_full=make_minimal_dynamic_fault_bc(),
        dynamic=True,
        dynamic_density=1.101,
        bulk_mu_v=0.0,
        bulk_lam_v=0.0,
        eta_fluid=2.0e-18 / 3600.0,
    )
    prestress = make_fault_prestress_patch_callbacks(
        preview["contact_plot"]["contact_s"],
        normal_prestress=6.6,
        mu_friction=0.6,
        background_ratio=0.98,
        patch_ratio=1.01,
        patch_half_width=0.03,
    )
    ctx = build_demo_contact_system(
        mu_friction=0.6,
        initial_gap_phys=0.0,
        reverse_gap_sign=False,
        rho_g=0.0,
        n_elem=8,
        element_type="tri",
        crack_theta=np.pi / 2.0,
        crack_x0=0.0,
        crack_y0=0.0,
        crack_length=0.6,
        bc_full=make_minimal_dynamic_fault_bc(),
        dynamic=True,
        dynamic_density=1.101,
        bulk_mu_v=0.0,
        bulk_lam_v=0.0,
        eta_fluid=2.0e-18 / 3600.0,
        contact_backend_opts={
            "rho_n": 1.0,
            "rho_t": 1.0,
            "gap_tol": 1.0e-12,
            "offset_coupling_mode": "incremental_reference",
            "get_s0": prestress["get_s0"],
            "get_w0": prestress["get_w0"],
        },
    )
    assert ctx["auto_reaction_nl_atol"] is not None
    assert ctx["reaction_nl_atol"] is not None
    assert ctx["reaction_nl_atol"] == ctx["auto_reaction_nl_atol"]
    assert 0.0 < ctx["reaction_nl_atol"] < 1.0e-4


def test_trim_failed_fixed_step_history_discards_failed_attempt():
    t_vals = np.array([0.0, 0.25, 0.5], dtype=float)
    y_vals = np.array([[0.0], [1.0], [9.0]], dtype=float)
    h_vals = np.array([0.25, 0.25, 0.25], dtype=float)
    error_estimates = [(1.0e-6, True, 2), (3.0, False, 30)]

    t_keep, y_keep, h_keep, err_keep, failure = _trim_failed_fixed_step_history(
        t_vals, y_vals, h_vals, error_estimates
    )

    np.testing.assert_allclose(t_keep, [0.0, 0.25])
    np.testing.assert_allclose(y_keep[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(h_keep, [0.25, 0.25])
    assert err_keep == [(1.0e-6, True, 2)]
    assert failure == {
        "step_index": 1,
        "solver_error": 3.0,
        "iterations": 30,
    }


def test_time_wrappers_expose_eta_fluid_keyword():
    assert "eta_fluid" in inspect.signature(run_one_step_case).parameters
    assert "eta_fluid" in inspect.signature(run_time_history_case).parameters


def test_prestressed_history_marks_incomplete_adaptive_run_as_failure(monkeypatch):
    class DummyPoro:
        def get_scales(self):
            return (2.0,)

    class DummyCS:
        def __init__(self):
            self.rhs = lambda y, t=None: np.zeros_like(np.asarray(y, dtype=float))
            self.y0 = np.array([0.0], dtype=float)
            self.projection = lambda y, t=None: np.asarray(y, dtype=float)
            self.A = None
            self.component_slices = None

    ctx = {
        "poro": DummyPoro(),
        "cs": DummyCS(),
        "time_method": "backward_euler",
        "solver_opts_contact": {},
        "integrator_opts_contact": {},
        "nl_atol_contact": [1.0e-6],
        "adaptive_opts_contact": {},
        "contact_plot": {
            "contact_coords": np.zeros((1, 2), dtype=float),
            "contact_s": np.zeros(1, dtype=float),
            "contact_perm": np.zeros(1, dtype=int),
            "crack_a": np.zeros(2, dtype=float),
            "crack_b": np.ones(2, dtype=float),
            "crack_tangent": np.array([1.0, 0.0], dtype=float),
            "crack_angle_deg": 0.0,
        },
    }

    monkeypatch.setattr(helper_mod, "build_prestressed_fault_dynamic_context", lambda **kwargs: ctx)
    monkeypatch.setattr(
        helper_mod,
        "audit_state",
        lambda y, ctx: {
            "gap_phys": np.zeros(1, dtype=float),
            "slip_t": np.zeros(1, dtype=float),
            "p_n": np.zeros(1, dtype=float),
            "p_t": np.zeros(1, dtype=float),
        },
    )
    monkeypatch.setattr(
        helper_mod,
        "evaluate_contact_offset_history",
        lambda y, t, ctx: (np.zeros(1, dtype=float), np.zeros(1, dtype=float)),
    )
    monkeypatch.setattr(
        helper_mod,
        "_compute_step_residual",
        lambda y_prev, y_new, t_new, h_step, ctx: np.zeros(1, dtype=float),
    )
    monkeypatch.setattr(
        helper_mod.solve_nivp,
        "solve_nivp",
        lambda **kwargs: (
            np.array([0.0, 0.4], dtype=float),
            np.array([[0.0], [1.0]], dtype=float),
            np.array([0.4, 0.4], dtype=float),
            None,
            [(1.0e-6, True, 2)],
            {
                "accepted": [True, False],
                "status": ["classic_accept", "minimum_step_reached"],
                "error": [1.0e-6, 3.0],
            },
        ),
    )

    history = run_prestressed_fault_dynamic_history_case(
        t_end_hours=2.0,
        n_steps=10,
        adaptive=True,
        return_attempts=True,
    )

    assert history.success is False
    assert history.terminated_early is True
    assert history.failure_status == "minimum_step_reached"
    assert history.failure_solver_error == 3.0
    assert history.failure_iterations is None
    assert history.times_hours[-1] == 0.8
