"""Regression tests for the cancellation-free small Jordan eigenvalue in the SOC
Fischer-Burmeister kernel (``solve_nivp.soc`` and the numba mirror).

At a complementary *slip* solution ``x`` and ``y`` sit on opposite cone-boundary
rays, so ``w = x**2 + y**2`` is rank-1 and its small eigenvalue
``lam1 = w_N - ||w_T||`` -> 0.  For a contact reaction ``||y|| ~ 1e6``, ``w_N`` and
``||w_T||`` are each ``~1e12`` and the direct difference cancels to the
``sqrt(eps)`` floor, so the FB root sits ``~1e-9`` (relative) off the cone.
Forming ``lam1 = (w_N**2 - ||w_T||**2)/(w_N + ||w_T||)`` with the numerator as a
Jordan-algebra sum of squares removes the cancellation.  These tests pin the
behaviour (they FAIL on the naive ``w_N - ||w_T||`` formula).
"""
import math

import numpy as np
import pytest

from solve_nivp.soc import soc_fb_phi, soc_fb_phi_and_jac, soc_fb_phi_and_jac_2d


def test_fb_phi_zero_on_cone_boundary_at_reaction_scale():
    # y on partial K boundary at 1e6 (reaction scale); x = slow complementary slip
    # on the opposite ray.  phi(x, y) == 0 analytically; the naive kernel returns
    # ~1e-2 here (sqrt(eps)*scale), the cancellation-free one ~1e-10.
    Y = 1.0e6
    for c in (1.0e-3, 1.0e-6, 1.0):
        x = np.array([c, -c])
        y = np.array([Y, Y])
        phi = soc_fb_phi(x, y)
        assert np.linalg.norm(phi) < 1.0e-6, (c, phi)
        # relative to the reaction scale this is machine precision
        assert np.linalg.norm(phi) / Y < 1.0e-12


def test_fb_phi_zero_for_x_in_cone_y_zero_large_scale():
    # phi(x, 0) = x - sqrt(x**2) = 0 for x in K, at large scale (d = 2 and d = 3).
    for x in (np.array([1.0e6, 1.0e6 - 1.0e-3]),
              np.array([1.0e6, 7.0e5, 7.0e5 - 1.0e-3])):
        y = np.zeros_like(x)
        phi = soc_fb_phi(x, y)
        assert np.linalg.norm(phi) / np.linalg.norm(x) < 1.0e-13


def test_fb_lam1_matches_exact_rational_2d():
    # The kernel helper forms lam1 = w_N - |w_T| via a sum-of-squares numerator.
    # Compare it directly (not reconstructed through phi, which itself cancels)
    # against the exact rational numerator over the identical float denominator:
    # this isolates the formula, which must be machine-accurate for d = 2.
    from fractions import Fraction as Fr

    from solve_nivp.soc import _fb_sqrt_lam1_2d

    rng = np.random.default_rng(3)
    worst_fix = worst_naive = 0.0
    for _ in range(2000):
        pN = rng.uniform(1.0e5, 3.0e6)
        mu = rng.uniform(0.2, 0.5)
        slack = 10.0 ** rng.uniform(-10, -3) * pN
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        x0 = abs(mu * (uT := 10.0 ** rng.uniform(-6, -2))) + 10.0 ** rng.uniform(-8, -3)
        x1 = mu * uT
        y0 = mu * pN
        y1 = sgn * (mu * pN - slack)
        w_N = x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1
        w_T_norm = abs(2.0 * (x0 * x1 + y0 * y1))
        den = w_N + w_T_norm
        sqrt_lam1 = _fb_sqrt_lam1_2d(x0, x1, y0, y1, w_N, w_T_norm)
        # exact numerator = w_N**2 - |w_T|**2  (rational), same float denominator
        X0, X1, Y0, Y1 = (Fr(v) for v in (x0, x1, y0, y1))
        WN = X0 * X0 + X1 * X1 + Y0 * Y0 + Y1 * Y1
        W1 = 2 * (X0 * X1 + Y0 * Y1)
        lam1_ref = float(WN * WN - W1 * W1) / den
        if lam1_ref > 0:
            ref = math.sqrt(lam1_ref)
            worst_fix = max(worst_fix, abs(sqrt_lam1 - ref) / ref)
            naive = math.sqrt(max(w_N - w_T_norm, 0.0))
            worst_naive = max(worst_naive, abs(naive - ref) / ref)
    assert worst_fix < 1.0e-13, worst_fix
    # sanity: the naive difference is catastrophic on these inputs (guards the test)
    assert worst_naive > 1.0e-3, worst_naive


def test_2d_fast_path_and_generic_agree_after_fix():
    # The closed-form d=2 branch and the generic spectral path must still agree
    # (both now cancellation-free), including on the hard near-boundary inputs.
    for vec in (np.array([1.0e-3, -1.0e-3, 1.0e6, 1.0e6]),
                np.array([0.3, 0.3, 0.4, 0.4]),
                np.array([1.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0])):
        x2, y2 = vec[:2], vec[2:]
        x3 = np.array([x2[0], x2[1], 0.0])
        y3 = np.array([y2[0], y2[1], 0.0])
        phi2, dx2, dy2 = soc_fb_phi_and_jac_2d(x2, y2)
        phi3, dx3, dy3 = soc_fb_phi_and_jac(x3, y3)
        np.testing.assert_allclose(phi2, phi3[:2], rtol=1e-10, atol=1e-9)


def test_fb_phi_zero_on_3d_cone_boundary_generic_direction():
    # d = 3: complementary boundary pair (x anti-parallel to y, both on the cone
    # boundary) at reaction scale, generic tangential direction.  phi == 0
    # analytically; the fix holds it to machine precision (naive gives ~sqrt(eps)).
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(3000):
        th = rng.uniform(0.0, 2.0 * math.pi)
        Y = rng.uniform(1.0e5, 3.0e6)
        dvec = np.array([math.cos(th), math.sin(th)])
        yT = Y * dvec
        y0 = math.sqrt(float(yT @ yT))          # y on the cone boundary
        c = 10.0 ** rng.uniform(-9, -3)
        xT = -c * dvec
        x0 = math.sqrt(float(xT @ xT))          # x on the opposite (polar) ray
        x = np.concatenate(([x0], xT))
        y = np.concatenate(([y0], yT))
        worst = max(worst, np.linalg.norm(soc_fb_phi(x, y)) / y0)
    assert worst < 1.0e-13, worst


def test_3d_cone_solve_feasible_to_machine():
    # A real 3-D De Saxce cone SSN solve (normal + 2 tangential) must hold the
    # converged reaction on the friction cone to machine precision (absolute).
    import scipy.sparse as sp

    from solve_nivp.moreau_jean_fremond import (
        DescriptorMoreauJeanFremondStepper,
        solve_mjf_adaptive,
    )

    mu, N0 = 0.3, 1.0e6
    A = sp.eye(3, format="csr")
    f = np.array([0.0, 5.0e5, 3.0e5])          # tangential drive -> slides
    tau = mu * N0 / math.hypot(1.0, 1.0)
    st = DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=lambda t, y: f.copy(),
        rhs_jac_callable=lambda t, y: sp.csr_matrix((3, 3)),
        D_extract=np.eye(3), B=np.eye(3),
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1, 2],
                   "mu_init": mu, "e": 0.0}],
        contact_offset_force=lambda y, t: np.array([N0, -tau, -tau]),
        theta=0.5, aux_law="constant", contact_solver="petsc_ssn",
        contact_linear_solver="dense", contact_residual="soc_fb",
        contact_ssn_tol=1.0e-14, theta_linear_solver="scipy",
    )
    slack, regime = [], []
    orig = st._solve_contact_petsc_ssn

    def wrap(W, b, mu_vec, shift_fn, p0, block_slices=None):
        pe, info, diag = orig(W, b, mu_vec, shift_fn, p0, block_slices=block_slices)
        slack.append(float(mu_vec[0]) * float(pe[0]) - float(np.hypot(pe[1], pe[2])))
        regime.append(info.regime[0] if info.regime else "?")
        return pe, info, diag

    st._solve_contact_petsc_ssn = wrap
    solve_mjf_adaptive(
        st, (0.0, 2.0), np.zeros(3), aux0={"mu": np.array([mu])},
        error_mask=np.array([False, True, True]), rtol=1e-4,
        atol=1e-6 * np.ones(3), h0=1e-2, h_max=0.2,
    )
    slack = np.abs(np.array(slack))
    regime = np.array(regime)
    slip = slack[regime == "slip"]
    assert slip.size > 0
    # absolute cone slack on a ~1e6 reaction is at the machine floor
    assert np.max(slip) < 1.0e-8, np.max(slip)
    assert np.median(slip) < 1.0e-10, np.median(slip)


def test_numba_kernel_matches_python_after_fix():
    numba_accel = pytest.importorskip("solve_nivp._numba_accel")
    if not hasattr(numba_accel, "_soc_fb_phi_jac_nb"):
        pytest.skip("numba kernel unavailable")
    from solve_nivp._numba_accel import _soc_fb_phi_jac_nb
    for x, y in ((np.array([1.0e-3, -1.0e-3]), np.array([1.0e6, 1.0e6])),
                 (np.array([1.0e6, 7.0e5, 7.0e5 - 1.0e-3]), np.zeros(3))):
        phi_py = soc_fb_phi(x, y)
        phi_nb, _, _ = _soc_fb_phi_jac_nb(x, y, False, 1.0e-14)
        np.testing.assert_allclose(phi_nb, phi_py, rtol=1e-9, atol=1e-8)
