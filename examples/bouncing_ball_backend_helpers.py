from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import solve_nivp
from solve_nivp.contact import build_impulse_contact

from spring_slider_coulomb_backend_helpers import _rattle_local_slider_backend


@dataclass(frozen=True)
class BouncingBallCase:
    mass: float = 1.0
    gravity: float = 9.81
    restitution: float = 0.8
    height0: float = 1.0
    vy0: float = 0.0
    x0: float = 0.0
    vx0: float = 0.0
    t_end: float = 2.0
    title: str = "Bouncing Ball"


def make_case(**kwargs: Any) -> dict[str, Any]:
    case = BouncingBallCase(**kwargs)
    return {
        "mass": float(case.mass),
        "gravity": float(case.gravity),
        "restitution": float(case.restitution),
        "height0": float(case.height0),
        "vy0": float(case.vy0),
        "x0": float(case.x0),
        "vx0": float(case.vx0),
        "t_end": float(case.t_end),
        "title": str(case.title),
    }


def make_persistent_contact_case(
    *,
    mass: float = 1.0,
    gravity: float = 9.81,
    t_end: float = 1.0,
    title: str = "Bouncing Ball Persistent Contact",
) -> dict[str, Any]:
    return make_case(
        mass=mass,
        gravity=gravity,
        restitution=0.0,
        height0=0.0,
        vy0=0.0,
        x0=0.0,
        vx0=0.0,
        t_end=t_end,
        title=title,
    )


def make_free_fall_case(
    *,
    mass: float = 1.0,
    gravity: float = 9.81,
    restitution: float = 0.8,
    height0: float = 1.0,
    vy0: float = 0.0,
    t_end: float = 2.0,
    title: str = "Bouncing Ball Free Fall",
) -> dict[str, Any]:
    return make_case(
        mass=mass,
        gravity=gravity,
        restitution=restitution,
        height0=height0,
        vy0=vy0,
        x0=0.0,
        vx0=0.0,
        t_end=t_end,
        title=title,
    )


def _analytical_reference(case: dict[str, Any], *, n_ref: int = 40001) -> dict[str, np.ndarray]:
    m = float(case["mass"])
    g = float(case["gravity"])
    e = float(case["restitution"])
    y0 = float(case["height0"])
    v0 = float(case["vy0"])
    t_end = float(case["t_end"])
    t = np.linspace(0.0, t_end, int(n_ref), dtype=float)

    q = np.zeros_like(t)
    v = np.zeros_like(t)
    r_n = np.full_like(t, np.nan)
    first_impact_time = np.nan
    first_rebound_apex = np.nan

    if y0 <= 1.0e-14 and abs(v0) <= 1.0e-14:
        q.fill(0.0)
        v.fill(0.0)
        r_n.fill(m * g)
        return {
            "t": t,
            "q": q,
            "v": v,
            "r_n": r_n,
            "impact_time": 0.0,
            "rebound_apex": 0.0,
        }

    t_cur = 0.0
    q_cur = y0
    v_cur = v0
    i0 = 0
    while i0 < t.size:
        if q_cur <= 1.0e-14 and abs(v_cur) <= 1.0e-14:
            mask = t >= t_cur
            q[mask] = 0.0
            v[mask] = 0.0
            r_n[mask] = m * g
            break

        disc = v_cur * v_cur + 2.0 * g * max(q_cur, 0.0)
        dt_hit = (v_cur + np.sqrt(max(disc, 0.0))) / g if g > 0.0 else np.inf
        if dt_hit < 0.0:
            dt_hit = np.inf
        t_hit = t_cur + dt_hit

        if not np.isfinite(t_hit) or t_hit > t_end:
            mask = t >= t_cur
            tau = t[mask] - t_cur
            q[mask] = q_cur + v_cur * tau - 0.5 * g * tau * tau
            v[mask] = v_cur - g * tau
            r_n[mask] = 0.0
            break

        mask = (t >= t_cur) & (t < t_hit)
        tau = t[mask] - t_cur
        q[mask] = q_cur + v_cur * tau - 0.5 * g * tau * tau
        v[mask] = v_cur - g * tau
        r_n[mask] = 0.0

        v_minus = v_cur - g * dt_hit
        v_plus = -e * v_minus
        if not np.isfinite(first_impact_time):
            first_impact_time = float(t_hit)
            first_rebound_apex = float(v_plus * v_plus / (2.0 * g)) if g > 0.0 else np.nan
        t_cur = t_hit
        q_cur = 0.0
        v_cur = v_plus
        if abs(v_plus) <= 1.0e-14:
            mask = t >= t_cur
            q[mask] = 0.0
            v[mask] = 0.0
            r_n[mask] = m * g
            break

        i0 = int(np.searchsorted(t, t_cur, side="left"))

    return {
        "t": t,
        "q": q,
        "v": v,
        "r_n": r_n,
        "impact_time": first_impact_time,
        "rebound_apex": first_rebound_apex,
    }


def _build_impulse_ball_problem(case: dict[str, Any]):
    mass = float(case["mass"])
    gravity = float(case["gravity"])
    restitution = float(case["restitution"])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(_t, y):
        v = np.asarray(y[:2], dtype=float)
        return np.array([0.0, -mass * gravity, v[0], v[1]], dtype=float)

    def rhs_jac(_t, _y, _Fk=None):
        J = np.zeros((4, 4), dtype=float)
        J[2, 0] = 1.0
        J[3, 1] = 1.0
        return J

    def gap_func(y, _t):
        return np.array([float(y[3])], dtype=float)

    y0 = np.array(
        [
            float(case["vx0"]),
            float(case["vy0"]),
            float(case["x0"]),
            float(case["height0"]),
        ],
        dtype=float,
    )
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.0, e=restitution)]
    start_in_contact = bool(float(case["height0"]) <= 1.0e-14 and abs(float(case["vy0"])) <= 1.0e-14)
    cs = build_impulse_contact(
        A,
        rhs,
        y0,
        contacts,
        gap_func,
        theta=1.0,
        component_slices=[slice(0, 2), slice(2, 4)],
        gap_tol=1.0e-12,
        retain_compressive_active=start_in_contact,
        activate_on_candidate_normal=start_in_contact,
        activity_tol=1.0e-12 if start_in_contact else 0.0,
        rhs_jac=rhs_jac,
    )
    return cs


def _estimate_impact_time(
    times: np.ndarray,
    q: np.ndarray,
    *,
    reactions: np.ndarray | None = None,
    reaction_tol: float = 1.0e-12,
    tol: float = 1.0e-8,
) -> float:
    if reactions is not None:
        r = np.asarray(reactions, dtype=float)
        idx_r = np.flatnonzero(np.abs(r) > reaction_tol)
        if idx_r.size:
            return float(np.asarray(times, dtype=float)[int(idx_r[0])])
    q = np.asarray(q, dtype=float)
    t = np.asarray(times, dtype=float)
    idx = np.flatnonzero(q <= tol)
    if idx.size == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    q0 = float(q[i - 1])
    q1 = float(q[i])
    t0 = float(t[i - 1])
    t1 = float(t[i])
    if abs(q1 - q0) <= 1.0e-14:
        return t1
    alpha = (tol - q0) / (q1 - q0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return t0 + alpha * (t1 - t0)


def _estimate_first_rebound_apex(times: np.ndarray, q: np.ndarray, *, impact_time: float) -> float:
    if not np.isfinite(impact_time):
        return np.nan
    t = np.asarray(times, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = t >= impact_time
    if not np.any(mask):
        return np.nan
    q_post = q[mask]
    if q_post.size == 0:
        return np.nan
    return float(np.max(q_post))


def run_backend_case(
    case: dict[str, Any],
    backend: str,
    *,
    n_steps: int = 2000,
    solver_max_iter: int = 200,
) -> dict[str, Any]:
    t0_wall = perf_counter()
    if backend in {"backward_euler", "sdirk2"}:
        cs = _build_impulse_ball_problem(case)
        t, y, h, _fk, info = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, float(case["t_end"])),
            y0=cs.y0,
            A=cs.A,
            method=backend,
            projection=cs.projection,
            solver="semismooth_newton",
            projection_opts={},
            solver_opts={
                "tol": 1.0e-12,
                "max_iter": int(solver_max_iter),
                "lam_update_strategy": "none",
                "globalization": "linesearch",
                "linear_solver": "splu",
            },
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=float(case["t_end"]) / float(n_steps),
            store_fk=False,
        )
        states = np.asarray(y[:, : cs.n_phys], dtype=float)
        reactions = np.asarray(y[:, cs.n_phys :], dtype=float)
        step_success = np.array([bool(item[1]) for item in info], dtype=bool) if len(info) else np.zeros(0, dtype=bool)
        step_iterations = np.array([int(item[2]) for item in info], dtype=int) if len(info) else np.zeros(0, dtype=int)
        def _err_scalar(value):
            if value is None:
                return np.nan
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 0:
                return np.nan
            return float(np.max(np.abs(arr)))

        step_solver_error = np.array([_err_scalar(item[0]) for item in info], dtype=float) if len(info) else np.zeros(0, dtype=float)
        failure_status = None
    elif backend == "rattle":
        slider_case = {
            "model_kind": "local_slider",
            "mass": float(case["mass"]),
            "mass_t": float(case["mass"]),
            "mass_n": float(case["mass"]),
            "stiffness": 0.0,
            "damping": 0.0,
            "normal_force": float(case["mass"]) * float(case["gravity"]),
            "mu_friction": 0.0,
            "q0": float(case["x0"]),
            "v0": float(case["vx0"]),
            "qn0": float(case["height0"]),
            "vn0": float(case["vy0"]),
            "t_end": float(case["t_end"]),
            "seed_initial_contact": bool(float(case["height0"]) <= 1.0e-14 and abs(float(case["vy0"])) <= 1.0e-14),
            "restitution": float(case["restitution"]),
        }
        res = _rattle_local_slider_backend(slider_case, n_steps=n_steps, solver_variant_overrides=None)
        t = np.asarray(res["times"], dtype=float)
        h = np.asarray(res["step_sizes"], dtype=float)
        states = np.asarray(res["states"], dtype=float)
        reactions = np.asarray(res["reactions"], dtype=float)
        info = list(res["step_info"])
        step_success = np.asarray(res["step_success"], dtype=bool)
        step_iterations = np.asarray(res["step_iterations"], dtype=int)
        step_solver_error = np.asarray(res["step_solver_error"], dtype=float)
        failure_status = res.get("failure_status")
    else:
        raise ValueError(f"Unsupported bouncing-ball backend {backend!r}")

    wall_time = perf_counter() - t0_wall
    ref = _analytical_reference(case)
    qy = np.asarray(states[:, 3], dtype=float)
    vy = np.asarray(states[:, 1], dtype=float)
    rn = reactions[:, 0] if reactions.shape[1] >= 1 else np.zeros_like(qy)
    qy_ref = np.interp(t, ref["t"], ref["q"])
    vy_ref = np.interp(t, ref["t"], ref["v"])
    impact_time = _estimate_impact_time(t, qy, reactions=rn, reaction_tol=1.0e-10)
    impact_time_ref = float(ref.get("impact_time", np.nan))
    if not np.isfinite(impact_time_ref):
        impact_time_ref = _estimate_impact_time(ref["t"], ref["q"], reactions=ref["r_n"], reaction_tol=1.0e-10)
    apex = _estimate_first_rebound_apex(t, qy, impact_time=impact_time)
    apex_ref = float(ref.get("rebound_apex", np.nan))
    if not np.isfinite(apex_ref):
        apex_ref = _estimate_first_rebound_apex(ref["t"], ref["q"], impact_time=impact_time_ref)

    backend_label = {
        "backward_euler": "Backward Euler + SOC",
        "sdirk2": "SDIRK2 + SOC",
        "rattle": "RATTLE (Lobatto IIIA-IIIB)",
    }[backend]

    return {
        "backend": backend,
        "backend_label": backend_label,
        "case": dict(case),
        "times": np.asarray(t, dtype=float),
        "states": np.asarray(states, dtype=float),
        "reactions": np.asarray(reactions, dtype=float),
        "step_sizes": np.asarray(h, dtype=float),
        "step_info": info,
        "step_success": step_success,
        "step_iterations": step_iterations,
        "step_solver_error": step_solver_error,
        "failure_status": failure_status,
        "q_y": qy,
        "v_y": vy,
        "r_n": rn,
        "q_y_ref": qy_ref,
        "v_y_ref": vy_ref,
        "impact_time": impact_time,
        "impact_time_ref": impact_time_ref,
        "rebound_apex": apex,
        "rebound_apex_ref": apex_ref,
        "summary": {
            "success": bool(np.all(step_success)) if step_success.size else False,
            "n_steps_completed": int(len(t) - 1),
            "n_failed_steps": int(np.count_nonzero(~step_success)) if step_success.size else 0,
            "last_step_success": bool(step_success[-1]) if step_success.size else False,
            "last_step_iterations": int(step_iterations[-1]) if step_iterations.size else None,
            "mean_step_iterations": float(np.mean(step_iterations)) if step_iterations.size else np.nan,
            "last_step_solver_error": float(step_solver_error[-1]) if step_solver_error.size else np.nan,
            "failure_status": failure_status,
            "max_abs_q_error": float(np.max(np.abs(qy - qy_ref))),
            "max_abs_v_error": float(np.max(np.abs(vy - vy_ref))),
            "max_penetration": float(np.max(np.maximum(-qy, 0.0))),
            "impact_time": float(impact_time) if np.isfinite(impact_time) else np.nan,
            "impact_time_ref": float(impact_time_ref) if np.isfinite(impact_time_ref) else np.nan,
            "impact_time_error": float(abs(impact_time - impact_time_ref)) if np.isfinite(impact_time) and np.isfinite(impact_time_ref) else np.nan,
            "rebound_apex": float(apex) if np.isfinite(apex) else np.nan,
            "rebound_apex_ref": float(apex_ref) if np.isfinite(apex_ref) else np.nan,
            "rebound_apex_error": float(abs(apex - apex_ref)) if np.isfinite(apex) and np.isfinite(apex_ref) else np.nan,
            "wall_time_s": float(wall_time),
            "final_qy": float(qy[-1]),
            "final_vy": float(vy[-1]),
            "final_rn": float(rn[-1]) if rn.size else np.nan,
        },
    }


def run_case_bundle(
    case: dict[str, Any],
    *,
    solvers: tuple[str, ...] = ("backward_euler", "sdirk2", "rattle"),
    n_steps: int = 2000,
    solver_max_iter: int = 200,
) -> dict[str, dict[str, Any]]:
    return {
        solver: run_backend_case(
            case,
            solver,
            n_steps=n_steps,
            solver_max_iter=solver_max_iter,
        )
        for solver in solvers
    }


def results_summary_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results.values():
        case = result["case"]
        row = {
            "backend": result["backend"],
            "backend_label": result["backend_label"],
            "mass": float(case["mass"]),
            "gravity": float(case["gravity"]),
            "restitution": float(case["restitution"]),
            "height0": float(case["height0"]),
            "vy0": float(case["vy0"]),
        }
        row.update(result["summary"])
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_sweep_dataframe(
    case: dict[str, Any],
    *,
    solvers: tuple[str, ...] = ("backward_euler", "sdirk2", "rattle"),
    step_counts: tuple[int, ...] = (250, 500, 1000, 2000),
    solver_max_iter: int = 200,
) -> pd.DataFrame:
    if isinstance(step_counts, (int, np.integer)):
        step_counts = (int(step_counts),)
    rows = []
    for n_steps in step_counts:
        results = run_case_bundle(case, solvers=solvers, n_steps=int(n_steps), solver_max_iter=solver_max_iter)
        for result in results.values():
            s = result["summary"]
            rows.append(
                {
                    "backend": result["backend"],
                    "backend_label": result["backend_label"],
                    "n_steps": int(n_steps),
                    "dt": float(case["t_end"]) / float(n_steps),
                    "success": bool(s["success"]),
                    "mean_step_iterations": float(s["mean_step_iterations"]),
                    "wall_time_s": float(s["wall_time_s"]),
                    "max_abs_q_error": float(s["max_abs_q_error"]),
                    "max_abs_v_error": float(s["max_abs_v_error"]),
                    "impact_time_error": float(s["impact_time_error"]),
                    "rebound_apex_error": float(s["rebound_apex_error"]),
                    "max_penetration": float(s["max_penetration"]),
                }
            )
    return pd.DataFrame(rows)
