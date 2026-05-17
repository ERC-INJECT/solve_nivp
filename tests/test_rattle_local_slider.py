"""Regression tests for the local spring-slider RATTLE helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

pytestmark = pytest.mark.examples

spring_helpers = pytest.importorskip(
    "spring_slider_coulomb_backend_helpers",
    reason="requires the spring-slider example helper module",
)
_rattle_local_slider_backend = spring_helpers._rattle_local_slider_backend
_rattle_schur_patch_backend = spring_helpers._rattle_schur_patch_backend
make_fem_scaled_chain_case = spring_helpers.make_fem_scaled_chain_case
make_multinode_chain_case = spring_helpers.make_multinode_chain_case
make_schur_patch_case = spring_helpers.make_schur_patch_case
run_case_bundle = spring_helpers.run_case_bundle

bouncing_helpers = pytest.importorskip(
    "bouncing_ball_backend_helpers",
    reason="requires the bouncing-ball example helper module",
)
convergence_sweep_dataframe = bouncing_helpers.convergence_sweep_dataframe
make_persistent_contact_case = bouncing_helpers.make_persistent_contact_case


def test_rattle_elastic_bounce_detects_repeated_impacts():
    case = {
        "model_kind": "local_slider",
        "mass": 1.0,
        "mass_t": 1.0,
        "mass_n": 1.0,
        "stiffness": 0.0,
        "damping": 0.0,
        "normal_force": 9.81,
        "mu_friction": 0.0,
        "q0": 0.0,
        "v0": 0.0,
        "qn0": 1.0,
        "vn0": 0.0,
        "t_end": 3.0,
        "seed_initial_contact": False,
        "restitution": 1.0,
    }

    result = _rattle_local_slider_backend(case, n_steps=3000, solver_variant_overrides=None)

    assert result["failure_status"] is None
    assert np.all(np.asarray(result["step_success"], dtype=bool))

    times = np.asarray(result["times"], dtype=float)
    states = np.asarray(result["states"], dtype=float)
    reactions = np.asarray(result["reactions"], dtype=float)
    q_n = states[:, 3]
    v_n = states[:, 1]
    r_n = reactions[:, 0]

    impact_indices = np.flatnonzero(np.abs(r_n) > 1.0e-8)
    assert impact_indices.size >= 3

    # The first two elastic rebounds should return to roughly the original height.
    apex_heights = []
    for idx in impact_indices[:2]:
        hi = min(len(q_n), int(idx) + 1000)
        j = int(idx + np.argmax(q_n[idx:hi]))
        apex_heights.append(float(q_n[j]))
        assert abs(float(v_n[j])) < 1.0e-10

    assert abs(apex_heights[0] - 1.0) < 5.0e-3
    assert abs(apex_heights[1] - 1.0) < 5.0e-3
    assert float(np.min(q_n)) >= -1.0e-12
    assert float(np.max(np.abs(v_n))) < 6.0


def test_bouncing_ball_convergence_sweep_accepts_single_int_step_count():
    case = make_persistent_contact_case(t_end=0.1)
    df = convergence_sweep_dataframe(
        case,
        solvers=("rattle",),
        step_counts=500,
        solver_max_iter=50,
    )
    assert len(df) == 1
    assert int(df.iloc[0]["n_steps"]) == 500


def test_rattle_multinode_chain_runs_and_stays_nonpenetrating():
    case = make_multinode_chain_case(
        n_nodes=3,
        tangential_stiffness=10.0,
        tangential_coupling=4.0,
        tangential_damping=0.5,
        normal_stiffness=0.0,
        normal_force_profile=[8.0, 10.0, 12.0],
        mu_friction=[0.2, 0.3, 0.4],
        v_t0=[1.0, 0.5, 0.0],
        q_n0=[0.0, 0.0, 0.0],
        seed_initial_contact=True,
        t_end=0.2,
        title="RATTLE Chain Regression",
    )
    results = run_case_bundle(
        case,
        solvers=("rattle",),
        n_steps=100,
        solver_max_iter=50,
    )
    result = results["rattle"]

    assert result["failure_status"] is None
    assert bool(result["summary"]["success"])
    assert float(result["summary"]["max_penetration"]) <= 1.0e-8
    assert np.asarray(result["states"], dtype=float).shape[1] == 12
    assert np.asarray(result["r_n_nodes"], dtype=float).shape[1] == 3


def test_rattle_fem_scaled_chain_notebook_case_converges():
    case = make_fem_scaled_chain_case(
        eta=1.05,
        mu_friction=0.1,
        sigma_n_expected=30.0,
        traction_ramp_duration_hours=None,
        n_elem=12,
        n_nodes=5,
        tangential_coupling_ratio=0.5,
        edge_stiffness_factor=0.7,
        t_end=1.0,
        seed_initial_contact=True,
    )
    results = run_case_bundle(
        case,
        solvers=("rattle",),
        n_steps=800,
        solver_max_iter=50,
    )
    result = results["rattle"]

    assert result["failure_status"] is None
    assert bool(result["summary"]["success"])
    assert float(result["summary"]["max_penetration"]) <= 1.0e-8
    assert float(result["summary"]["max_coulomb_ratio"]) <= 1.0 + 1.0e-8


def test_rattle_fem_scaled_chain_seeds_initial_reaction_history():
    case = make_fem_scaled_chain_case(
        eta=1.05,
        mu_friction=0.1,
        sigma_n_expected=30.0,
        traction_ramp_duration_hours=None,
        n_elem=12,
        n_nodes=5,
        tangential_coupling_ratio=0.5,
        edge_stiffness_factor=0.7,
        t_end=1.0,
        seed_initial_contact=True,
    )
    result = run_case_bundle(
        case,
        solvers=("rattle",),
        n_steps=100,
        solver_max_iter=50,
    )["rattle"]

    rn0 = np.asarray(result["r_n_nodes"], dtype=float)[0]
    rt0 = np.asarray(result["r_t_nodes"], dtype=float)[0]
    np.testing.assert_allclose(rn0, 30.0, atol=1.0e-12)
    np.testing.assert_allclose(rt0, -3.0, atol=1.0e-12)


def test_rattle_fem_scaled_chain_eta_one_sticks_with_alpha_one():
    case = make_fem_scaled_chain_case(
        eta=1.0,
        mu_friction=0.1,
        sigma_n_expected=30.0,
        traction_ramp_duration_hours=None,
        n_elem=20,
        n_nodes=5,
        tangential_coupling_ratio=0.5,
        edge_stiffness_factor=0.7,
        t_end=1.5,
        seed_initial_contact=True,
    )
    result = run_case_bundle(
        case,
        solvers=("rattle",),
        n_steps=50,
        solver_max_iter=50,
        solver_variant_overrides={"rattle": {"prox_alpha": 1.0}},
    )["rattle"]

    assert result["failure_status"] is None
    assert bool(result["summary"]["success"])
    assert float(result["summary"]["max_penetration"]) <= 1.0e-8
    assert abs(float(result["summary"]["final_q"])) <= 1.0e-8
    assert abs(float(result["summary"]["final_v"])) <= 1.0e-8
    assert abs(float(result["summary"]["final_rn"]) - 30.0) <= 1.0e-8
    assert abs(float(result["summary"]["final_rt"]) + 3.0) <= 1.0e-6


def _dense_schur_patch_case(n_nodes: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mass_raw = rng.standard_normal((2 * n_nodes, 2 * n_nodes))
    mass_matrix = mass_raw @ mass_raw.T + 0.5 * np.eye(2 * n_nodes)
    return make_schur_patch_case(
        mass_matrix_contact=mass_matrix,
        n_nodes=n_nodes,
        tangential_stiffness=50.0,
        tangential_damping=2.0,
        normal_stiffness=0.0,
        normal_damping=0.0,
        normal_force_profile=5.0,
        tangential_force_profile=1.5,
        mu_friction=0.3,
        q_n0=0.0,
        t_end=0.25,
    )


def test_rattle_schur_patch_semismooth_newton_drives_gap_to_roundoff():
    case = _dense_schur_patch_case()
    n_nodes = int(case["n_nodes"])

    out_ssn = _rattle_schur_patch_backend(
        case,
        n_steps=200,
        solver_variant_overrides={"rattle": {"stage1_method": "semismooth_newton"}},
    )
    assert out_ssn["failure_status"] is None
    q_n_ssn = np.asarray(out_ssn["states"])[:, 2 * n_nodes :][:, 0::2]
    assert float(np.max(np.abs(q_n_ssn))) < 1.0e-13
    assert int(np.max(out_ssn["stage1_iterations"])) <= 10

    out_fp = _rattle_schur_patch_backend(
        case,
        n_steps=200,
        solver_variant_overrides={"rattle": {"stage1_method": "fixed_point"}},
    )
    assert out_fp["failure_status"] is None
    q_n_fp = np.asarray(out_fp["states"])[:, 2 * n_nodes :][:, 0::2]
    u_ssn = np.asarray(out_ssn["states"])[:, : 2 * n_nodes]
    u_fp = np.asarray(out_fp["states"])[:, : 2 * n_nodes]
    assert np.allclose(u_ssn, u_fp, atol=1.0e-6, rtol=1.0e-6)
    assert float(np.max(np.abs(q_n_fp))) < 1.0e-12
