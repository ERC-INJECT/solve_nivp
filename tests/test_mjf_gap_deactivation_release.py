"""Gap-deactivated (geometrically open) contacts must RELEASE their prestress.

A contact dropped from the cone complementarity problem by geometric gap
activation (``gap_callable``, ``g > gap_tol``) carries **zero total reaction**:
its effective impulse is ``p_effective = 0``, hence the increment applied to the
bulk is ``p_contact = p_effective - offset = -offset`` -- the prestress is
released, not retained.

The reduced active-subset solve books the inactive rows at the two sites

  * ``_solve_contact_active_subset`` (partial subset: some contacts open) and
  * the ``active_blocks.size == 0`` branch (all contacts open).

Both previously seeded the inactive rows with ``p_effective = offset`` (the
frozen prestress), so an open contact kept a phantom traction: it reported a
nonzero reaction power and, with a prestressed friction offset, pinned
``|r_t|/r_n`` at the prestress utilization instead of releasing.  These tests
exercise both sites; they are red against the retain-offset booking.
"""

import numpy as np

from solve_nivp.moreau_jean_fremond import DescriptorMoreauJeanFremondStepper

# --- shared parameters -------------------------------------------------------
H = 0.1
N0_A, N0_B, T0_B = 2.0, 3.0, 1.0    # prestress FORCES (per reaction DOF)
VN_B0, VT_B0 = 3.0, 1.0             # contact B: separating + sliding start
GAP_TOL = 1.0e-6


def _prestressed_pair(gap):
    """Two INDEPENDENT prestressed contacts (block-diagonal, A = I_5).

    State y = [qnA, vnA, qnB, vnB, vtB].
      * Contact A (reaction dof 0): normal-only, at rest -> stays CLOSED.
      * Contact B (reaction dofs 1 normal, 2 tangential): frictional, starts
        with a separating normal velocity and a sliding tangential velocity ->
        OPENS after the first step.
    The offset (prestress) is nonzero on every reaction dof.  ``gap=False`` drops
    the gap index set -> both contacts stay in the velocity CP (persistent).
    """
    A = np.eye(5)

    def rhs(t, y):
        return np.array([y[1], 0.0, y[3], 0.0, 0.0])

    def rhs_jac(t, y):
        J = np.zeros((5, 5))
        J[0, 1] = 1.0     # qnA' = vnA
        J[2, 3] = 1.0     # qnB' = vnB
        return J

    D = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],   # u_N,A = vnA
        [0.0, 0.0, 0.0, 1.0, 0.0],   # u_N,B = vnB
        [0.0, 0.0, 0.0, 0.0, 1.0],   # u_T,B = vtB
    ])
    B = np.array([
        [0.0, 0.0, 0.0],   # qnA
        [1.0, 0.0, 0.0],   # vnA  <- pN_A
        [0.0, 0.0, 0.0],   # qnB
        [0.0, 1.0, 0.0],   # vnB  <- pN_B
        [0.0, 0.0, 1.0],   # vtB  <- pT_B
    ])
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[
            {"vel_normal_idx": 0, "vel_tangential_idx": [], "mu": 0.0, "e": 0.0},
            {"vel_normal_idx": 1, "vel_tangential_idx": [2], "mu": 0.5, "e": 0.0},
        ],
        theta=0.5,
        contact_solver="pgs",
        theta_linear_solver="scipy",
        gap_callable=(lambda z: np.array([z[0], z[2]])) if gap else None,
        gap_tol=GAP_TOL,
        contact_offset_force=lambda y, t: np.array([N0_A, N0_B, T0_B]),
    )


def _prestressed_single(gap):
    """One prestressed frictional contact that separates -> exercises the
    ``active_blocks.size == 0`` (all-open) booking site.

    State y = [qn, vn, vt]; reaction dofs 0 (normal), 1 (tangential).
    """
    A = np.eye(3)

    def rhs(t, y):
        return np.array([y[1], 0.0, 0.0])

    def rhs_jac(t, y):
        J = np.zeros((3, 3))
        J[0, 1] = 1.0
        return J

    D = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1], "mu": 0.5, "e": 0.0}],
        theta=0.5,
        contact_solver="pgs",
        theta_linear_solver="scipy",
        gap_callable=(lambda z: np.array([z[0]])) if gap else None,
        gap_tol=GAP_TOL,
        contact_offset_force=lambda y, t: np.array([N0_B, T0_B]),
    )


def _march(stepper, y0, mu, nsteps):
    y = np.asarray(y0, dtype=float).copy()
    aux = {"mu": np.asarray(mu, dtype=float).copy()}
    trace = []
    for k in range(nsteps):
        y, aux, info = stepper.step(k * H, y, aux, H)
        trace.append((y.copy(), info))
    return trace


# ---------------------------------------------------------------------------
# (a) an OPEN contact books zero effective impulse and releases its prestress.
# ---------------------------------------------------------------------------
def test_gap_deactivated_contact_releases_prestress_partial_subset():
    y0 = np.array([0.0, 0.0, 0.0, VN_B0, VT_B0])
    trace = _march(_prestressed_pair(gap=True), y0, [0.0, 0.5], nsteps=4)

    off_B = np.array([H * N0_B, H * T0_B])   # prestress IMPULSE on B's rows
    B_rows = [1, 2]
    checked = 0
    for k, (y, info) in enumerate(trace):
        if info["regime"][1] != "inactive":
            continue                          # only the geometrically-open steps
        checked += 1
        p_eff_B = info["p_contact_effective"][B_rows]
        p_ctc_B = info["p_contact"][B_rows]
        # TOTAL reaction of an open contact is zero...
        np.testing.assert_allclose(
            p_eff_B, np.zeros(2), atol=1.0e-10,
            err_msg=f"open contact retains phantom effective impulse at step {k}: {p_eff_B}",
        )
        # ...so the increment handed to the bulk is the prestress RELEASE, -offset.
        np.testing.assert_allclose(
            p_ctc_B, -off_B, atol=1.0e-10,
            err_msg=f"open contact fails to release its prestress at step {k}: {p_ctc_B}",
        )
    assert checked >= 2, "test did not reach the gap-deactivated regime"


def test_gap_deactivated_contact_releases_prestress_all_open():
    # Single separating contact -> active_blocks.size == 0 booking site.
    trace = _march(_prestressed_single(gap=True), np.array([0.0, VN_B0, VT_B0]),
                   [0.5], nsteps=4)
    off = np.array([H * N0_B, H * T0_B])
    checked = 0
    for k, (y, info) in enumerate(trace):
        if info["regime"][0] != "inactive":
            continue
        checked += 1
        np.testing.assert_allclose(
            info["p_contact_effective"], np.zeros(2), atol=1.0e-10,
            err_msg=f"all-open booking retains phantom effective impulse at step {k}",
        )
        np.testing.assert_allclose(
            info["p_contact"], -off, atol=1.0e-10,
            err_msg=f"all-open booking fails to release prestress at step {k}",
        )
    assert checked >= 2


# ---------------------------------------------------------------------------
# (c) an OPEN contact contributes zero contact (reaction) power.
# ---------------------------------------------------------------------------
def test_gap_deactivated_contact_books_zero_reaction_power():
    y0 = np.array([0.0, 0.0, 0.0, VN_B0, VT_B0])
    trace = _march(_prestressed_pair(gap=True), y0, [0.0, 0.5], nsteps=4)
    B_rows = [1, 2]
    checked = 0
    for k, (y, info) in enumerate(trace):
        if info["regime"][1] != "inactive":
            continue
        checked += 1
        u_B = info["u_kt"][B_rows]
        p_eff_B = info["p_contact_effective"][B_rows]
        # The open contact is genuinely moving (non-vacuous check): with the
        # retain-offset bug u_B . offset would be a nonzero phantom power.
        assert np.linalg.norm(u_B) > 1.0e-3
        power = float(u_B @ p_eff_B)
        assert abs(power) < 1.0e-10, (
            f"open contact books nonzero reaction power {power:.3e} at step {k}"
        )
    assert checked >= 2


# ---------------------------------------------------------------------------
# (b) releasing one contact must NOT perturb the other (closed) contact:
#     the closed contact matches a persistent (no gap_callable) reference.
# ---------------------------------------------------------------------------
def test_closed_contact_matches_persistent_reference_when_other_opens():
    y0 = np.array([0.0, 0.0, 0.0, VN_B0, VT_B0])
    gap_trace = _march(_prestressed_pair(gap=True), y0, [0.0, 0.5], nsteps=4)
    ref_trace = _march(_prestressed_pair(gap=False), y0, [0.0, 0.5], nsteps=4)

    A_dofs = [0, 1]   # qnA, vnA
    # sanity: contact B really does open in the gap run
    assert any(info["regime"][1] == "inactive" for _, info in gap_trace)
    for (y_gap, _), (y_ref, _) in zip(gap_trace, ref_trace):
        np.testing.assert_allclose(
            y_gap[A_dofs], y_ref[A_dofs], atol=1.0e-12,
            err_msg="opening contact B perturbed the closed contact A",
        )


# ---------------------------------------------------------------------------
# Stronger invariant: a separating contact carries no reaction whether it is
# geometrically dropped or velocity-solved-and-separating, so the whole gap-run
# state matches the persistent reference.  (Red with the retain-offset bug,
# which freezes the open contact at its prestress and diverges.)
# ---------------------------------------------------------------------------
def test_gap_run_matches_persistent_reference_full_state():
    y0 = np.array([0.0, 0.0, 0.0, VN_B0, VT_B0])
    gap_end = _march(_prestressed_pair(gap=True), y0, [0.0, 0.5], nsteps=4)[-1][0]
    ref_end = _march(_prestressed_pair(gap=False), y0, [0.0, 0.5], nsteps=4)[-1][0]
    np.testing.assert_allclose(gap_end, ref_end, atol=1.0e-9)


# ===========================================================================
# COUPLED case: with a nonzero Delassus off-diagonal (W_AB != 0) the prestress
# RELEASE of an opening contact must couple -- within the SAME step -- into a
# still-closed neighbour's velocity CP.  When contact B opens it hands the bulk
# p_contact_B = -offset_B; through the off-diagonal this perturbs the closed
# contact A's velocity by W_AB * p_contact_B = -W_AB * offset_B.  The reduced
# active-subset solve must therefore build its rhs from the FULL prestress
# offset (b_a = (b_soccp - W @ offset)[active]) so the closed rows feel that
# release; the open rows stay OUT of the cone problem and keep p_effective = 0.
#
# The block-diagonal tests above cannot see this: with W_AB = 0 the closed
# contact is genuinely decoupled.  Here A and B share a Delassus off-diagonal,
# so if the reduced rhs zeroes the open row's offset the closed contact misses
# the -W_AB*offset_B kick and drifts INTO the wall at a steady O(h) normal
# velocity (theta*W_AB*h*f_B per step) -- interpenetration.
# ===========================================================================
C_COUPLE = 0.5                   # Delassus off-diagonal W_AB = W_BA
FA_C, FB_C = 2.0, 3.0            # prestress FORCES on the coupled pair
VB0_C = 3.0                     # contact B separating normal velocity


def _coupled_pair(gap):
    """Two normal-only prestressed contacts sharing a Delassus off-diagonal.

    State y = [qA, qB, vA, vB]; reactions p = [pN_A, pN_B] (both normal).  The
    reaction-to-velocity map couples the contacts through the shared velocity
    dofs:
        vA <- pN_A + C pN_B,   vB <- C pN_A + pN_B   =>   W = [[1, C], [C, 1]].
    Contact A (index 0) starts at rest and stays CLOSED; contact B (index 1)
    starts separating (vB0 > 0) and OPENS after the first step.  ``gap=False``
    keeps both in the velocity CP -> the persistent reference, which
    velocity-solves B (separating -> p_effective_B = 0) and therefore sees the
    release exactly through the full 2x2 Delassus.
    """
    A = np.eye(4)

    def rhs(t, y):
        return np.array([y[2], y[3], 0.0, 0.0])

    def rhs_jac(t, y):
        J = np.zeros((4, 4))
        J[0, 2] = 1.0     # qA' = vA
        J[1, 3] = 1.0     # qB' = vB
        return J

    D = np.array([
        [0.0, 0.0, 1.0, 0.0],   # u_N,A = vA
        [0.0, 0.0, 0.0, 1.0],   # u_N,B = vB
    ])
    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, C_COUPLE],        # vA <- pN_A + C pN_B
        [C_COUPLE, 1.0],        # vB <- C pN_A + pN_B
    ])
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[
            {"vel_normal_idx": 0, "vel_tangential_idx": [], "mu": 0.0, "e": 0.0},
            {"vel_normal_idx": 1, "vel_tangential_idx": [], "mu": 0.0, "e": 0.0},
        ],
        theta=0.5,
        contact_solver="pgs",
        theta_linear_solver="scipy",
        gap_callable=(lambda z: np.array([z[0], z[1]])) if gap else None,
        gap_tol=GAP_TOL,
        contact_offset_force=lambda y, t: np.array([FA_C, FB_C]),
    )


def test_coupled_closed_contact_holds_non_penetration_when_neighbor_opens():
    """The still-closed contact must maintain non-penetration when its coupled
    neighbour opens: zero normal velocity at every gap-deactivated step and no
    steady O(h) drift into the wall over the march."""
    y0 = np.array([0.0, 0.0, 0.0, VB0_C])
    trace = _march(_coupled_pair(gap=True), y0, [0.0, 0.0], nsteps=6)
    # contact B really does gap-deactivate (non-vacuous)
    assert any(info["regime"][1] == "inactive" for _, info in trace)
    checked = 0
    for k, (y, info) in enumerate(trace):
        if info["regime"][1] != "inactive":
            continue
        checked += 1
        u_A = float(info["u_kt"][0])          # normal velocity of the closed contact A
        assert abs(u_A) < 1.0e-8, (
            f"closed contact interpenetrates at O(h) velocity {u_A:.6e} at step {k} "
            f"(opening neighbour's release W_AB*offset_B did not couple in; "
            f"predicted miss theta*C*h*fB = {0.5 * C_COUPLE * H * FB_C:.6e})"
        )
    assert checked >= 2, "test did not reach the gap-deactivated regime"
    # no steady drift: the closed contact stays at the wall (qA ~ 0) end-of-march.
    qA_end = float(trace[-1][0][0])
    assert abs(qA_end) < 1.0e-8, (
        f"closed contact drifted into the wall over the march: qA={qA_end:.6e}"
    )


def test_coupled_closed_contact_reaction_reflects_neighbor_release():
    """The closed contact's reaction must reflect the neighbour's release: it
    matches the persistent (full-CP) reference, which velocity-solves the
    separating neighbour and therefore couples the release exactly."""
    y0 = np.array([0.0, 0.0, 0.0, VB0_C])
    gap_trace = _march(_coupled_pair(gap=True), y0, [0.0, 0.0], nsteps=6)
    ref_trace = _march(_coupled_pair(gap=False), y0, [0.0, 0.0], nsteps=6)
    assert any(info["regime"][1] == "inactive" for _, info in gap_trace)
    for k, ((y_g, ig), (y_r, ir)) in enumerate(zip(gap_trace, ref_trace)):
        np.testing.assert_allclose(
            ig["p_contact_effective"][0], ir["p_contact_effective"][0], atol=1.0e-8,
            err_msg=(
                f"closed contact reaction ignores neighbour release at step {k}: "
                f"gap={ig['p_contact_effective'][0]:.6e} ref={ir['p_contact_effective'][0]:.6e}"
            ),
        )
        np.testing.assert_allclose(
            y_g[[0, 2]], y_r[[0, 2]], atol=1.0e-8,
            err_msg=f"closed contact state diverges from persistent reference at step {k}",
        )
    # Guard against a vacuous match: the release genuinely LOADS contact A, so
    # its reaction is not merely its own prestress hold (offset impulse H*FA_C).
    max_load = max(
        abs(float(ig["p_contact_effective"][0]) - H * FA_C) for _, ig in ref_trace
    )
    assert max_load > 1.0e-3, "neighbour release did not measurably load contact A"
