"""Impulse-level contact formulation helpers.

Automates the augmentation of a smooth first-order system with contact
reaction DOFs, De Saxcé–Frémond reaction rows, and SOC projection.

The user provides:

* ``A``, ``rhs_smooth``, ``y0`` — the physical system (no contact).
* Per-contact: ``vel_normal_idx``, ``vel_tangential_idx``, ``mu``, ``e``.
* ``gap_func`` — signed gap on the *physical* state.
* Optionally ``B`` — coupling matrix (auto-generated if omitted).

The helper returns everything needed to call :func:`solve_ivp_ns`::

    cs = build_impulse_contact(A, rhs, y0, contacts, gap_func)
    t, y, *_ = solve_ivp_ns(
        fun=cs.rhs, y0=cs.y0, A=cs.A,
        projection=cs.projection,
        component_slices=cs.component_slices,
        integrator_opts=cs.integrator_opts,
        method='backward_euler', solver='semismooth_newton', ...
    )
    # Physical state: y[:, :cs.n_phys]
    # Reaction impulses: y[:, cs.n_phys:]
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .projections import (
    AlgebraicConstraintProjection,
    AnisotropicSOCProjection,
    CompositeContactProjection,
    MuScaledSOCProjection,
)


# ──────────────────────────────────────────────────────────────────────
# Return type
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ContactSystem:
    """Augmented system ready for ``solve_ivp_ns``.

    Attributes
    ----------
    A : ndarray or sparse
        Augmented descriptor matrix ``block_diag(A_phys, 0)``.
    rhs : callable
        Augmented RHS including physical forces, ``B p / h`` coupling,
        and De Saxcé–Frémond reaction rows.
    y0 : ndarray
        Augmented initial condition (reactions initialised to zero).
    projection : MuScaledSOCProjection
        SOC projector with ``zero_inactive=True`` and shifted block
        indices on the augmented state.
    component_slices : list of slice
        Physical slices extended with the reaction block.
    integrator_opts : dict
        ``{'pass_prev_state': True, 'pass_step_size': True}``
    n_phys : int
        Number of physical DOFs.  ``y[:n_phys]`` recovers the physical
        state from the augmented solution.
    B : ndarray or sparse
        Coupling matrix (provided or auto-generated).
    rhs_jac : callable or None
        Auto-generated Jacobian of the augmented RHS.  Composes the
        smooth-RHS Jacobian (user-provided or finite-difference) with
        the coupling and De Saxcé blocks.  Pass to the solver via
        ``solver_opts={'rhs_jac': cs.rhs_jac}``.
    """
    A: Any
    rhs: Callable
    y0: np.ndarray
    projection: MuScaledSOCProjection
    component_slices: List
    integrator_opts: Dict[str, Any]
    n_phys: int
    B: Any = None
    rhs_jac: Optional[Callable] = None


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────
def build_impulse_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    theta=1.0,
    coupling_theta=None,
    incremental_coupling=False,
    fremond_contact=False,
    component_slices=None,
    C_extract=None,
    D_extract=None,
    rate_form=False,
    constraints=None,
    get_s0=None,
    get_w0=None,
    get_ds0_dz=None,
    get_dw0_dz=None,
    get_B=None,
    step_size_ref=None,
    rhs_jac=None,
):
    r"""Build an augmented impulse-level system for frictional contact.

    Parameters
    ----------
    A : ndarray or sparse, shape ``(n_phys, n_phys)``
        Physical descriptor / mass matrix.
    rhs_smooth : callable
        Smooth RHS on the *physical* state only:
        ``rhs(t, y_phys) -> ndarray(n_phys)``.
        Must **not** include contact reactions.
    y0 : ndarray, shape ``(n_phys,)``
        Physical initial condition.
    contacts : list of dict
        Per-contact specification.  Each dict contains:

        * ``vel_normal_idx`` : int — physical DOF index for :math:`v_N`.
        * ``vel_tangential_idx`` : int or list of int — physical DOF
          index(es) for :math:`v_T`.
        * ``mu`` : float or callable — friction coefficient.  A callable
          has signature ``mu(y_phys) -> float``.
        * ``beta`` : float or callable, optional — dilatancy coefficient
          :math:`\beta = \tan\psi` (default 0).  Same calling convention
          as ``mu``.  Must satisfy :math:`0 \le \beta \le \mu`.
          The De Saxcé augmentation uses :math:`\alpha = \mu - \beta`.
        * ``e`` : float, optional — coefficient of restitution (default 0).

    gap_func : callable or None, optional
        Signed gap on the *physical* state:
        ``gap(y_phys, t) -> ndarray(n_contacts)``.
        Contact *k* is active when ``gap[k] <= 0``.
        If *None* and ``C_extract`` is provided, auto-generated from
        the normal rows of ``C_extract`` (gap = ``C @ y_phys`` at
        the normal indices).  Required when ``C_extract`` is None.
    B : ndarray or sparse, shape ``(n_phys, n_react)``, optional
        Coupling matrix mapping reaction unknowns to generalised forces.
        If *None* and ``C_extract`` is provided, auto-generated as
        ``C_extract.T`` (virtual-work pairing).
        If *None* and ``C_extract`` is also None, auto-generated from
        the ``vel_normal_idx`` / ``vel_tangential_idx`` mappings
        (identity coupling, valid for point-mass problems).
    theta : float, default 1.0
        Implicit blending coefficient for the **Frémond restitution**
        :math:`c = \theta(1+e)-1`.  This accounts for how the integrator
        blends old/new evaluations in the algebraic (contact) rows.

        * ``1.0`` — Backward Euler, or any method whose algebraic rows
          are fully implicit (SDIRK2 stages are each a BE solve).
        * ``0.5`` — Theta / Moreau midpoint.

    coupling_theta : float or None, default None
        Pre-compensation factor for the **reaction coupling** in the
        physical (momentum) rows: ``B r / (coupling_theta * h)``.
        After the integrator multiplies by its own implicit coefficient,
        the net impulse enters at full strength.

        If *None*, defaults to ``theta`` (correct for one-step
        :math:`\theta`-methods where the same :math:`\theta` scales both
        physical and algebraic rows).

        For SDIRK2, set ``coupling_theta = γ ≈ 0.293`` (the diagonal
        Butcher coefficient) while keeping ``theta = 1.0``.  Each stage
        divides by :math:`\gamma h`, so pre-dividing by :math:`\gamma`
        ensures the impulse enters at full strength.
    incremental_coupling : bool, default False
        When *True*, the reaction coupling in the physical rows uses the
        anti-blend formula

        .. math::
            B \bigl(r_{\text{prev}}
              + (r - r_{\text{prev}})\,/\,\theta_c\bigr) \,/\, h

        so that after :math:`\theta`-blending (old + new evaluations)
        the **net** impulse entering the momentum balance is exactly
        :math:`B\,r_{\text{new}}/h` — free of ghost impulses from the
        previous step's reactions.

        Set ``True`` for one-step :math:`\theta`-methods
        (:class:`ThetaMethod`, trapezoidal rule) where the old-state
        evaluation is blended into the residual.  Leave ``False``
        (default) for Backward Euler and multi-stage methods (SDIRK2)
        whose stages are individually fully implicit.
    fremond_contact : bool, default False
        When *True*, use the **Frémond θ-averaged** contact
        discretisation from Acary & Cadoux (JTCAM 2025, §3.2).

        Instead of evaluating the De Saxcé augmentation at the
        fully-implicit velocity :math:`v_{k+1}` (standard mode) or
        via naïve θ-blend-of-norms (plain ``ThetaMethod``), the
        augmentation is evaluated at the **θ-averaged velocity**
        :math:`v_{k+\theta} = \theta v_{k+1} + (1-\theta) v_k`,
        giving:

        .. math::
            \Theta_N = v_{N,k+\theta}
                     + (\theta(1{+}e) - 1)\, v_{N,k}
                     + \mu\|v_{T,k+\theta}\|

        This evaluates the **norm of the θ-blend** rather than
        the θ-blend of norms, guaranteeing positive contact
        dissipation (Proposition 1) provided

        .. math::
            \tfrac{1}{2} \le \theta \le \tfrac{1}{1 + \bar e}

        where :math:`\bar e = \max_\alpha e^\alpha`.  This
        condition is validated automatically and a ``ValueError``
        is raised if violated.

        Implies ``incremental_coupling=True`` for the physical
        coupling rows (impulse enters at full strength).

        For :math:`\theta = 1` (Backward Euler) the scheme
        reduces to the standard fully-implicit Moreau
        time-stepping (eq. 68–69 of the reference).
    component_slices : list of slice or list of array-like, optional
        Physical-level component partition.  Each element may be a
        ``slice`` object **or** a list / numpy array of integer DOF
        indices.  A reaction-DOF block is appended automatically.
    C_extract : ndarray or sparse, shape ``(n_contact, n_phys)``, optional
        Extraction operator mapping the full physical state to the
        contact-velocity (or contact-displacement) space.  When
        provided:

        * ``vel_normal_idx`` / ``vel_tangential_idx`` in each contact
          dict become **row indices of C_extract** (not physical-DOF
          indices).
        * De Saxcé augmentation operates in the extracted space.
        * ``B`` defaults to ``C_extract.T`` (virtual-work pairing).
        * ``gap_func`` defaults to ``C_extract @ y`` at the normal
          rows.

        When *None* (default), the current identity-extraction
        behaviour is preserved.
    D_extract : ndarray or sparse, same shape as ``C_extract``, optional
        Separate kinematic operator for **velocity** extraction (e.g.
        a dashpot-weighted evaluation operator ``D_σ``).  De Saxcé
        uses ``D_extract`` for the contact velocity while ``C_extract``
        is still used for gap detection.
        If *None*, defaults to ``C_extract``.
    rate_form : bool, default False
        When *True*, the contact velocity entering De Saxcé is
        computed as a backward-difference rate:

        .. math::
            v_c = D \frac{z - z_{\text{prev}}}{h}

        rather than ``D @ z`` directly.  Use this when the physical
        unknowns are **displacements** (not velocities).

        Requires ``θ(1+e) = 1`` for every contact (i.e. backward
        Euler with ``e = 0``) to avoid needing the previous-step
        velocity, which is not available in rate form.
    constraints : list of dict, optional
        Algebraic constraints on the *physical* DOFs, enforced
        simultaneously with the SOC frictional contact.  Each dict
        has the same keys as
        :class:`~solve_nivp.AlgebraicConstraintProjection`'s
        ``constraints`` parameter:

        * ``'g'`` — constraint map ``g(y_sub) -> q_sub``.
        * ``'dg_dy'`` — analytical Jacobian (optional; FD if omitted).
        * ``'y_slice'`` — slice locating the input DOFs in the
          *physical* state vector.
        * ``'q_slice'`` — slice locating the output DOFs in the
          *physical* state vector.

        When provided, a
        :class:`~solve_nivp.CompositeContactProjection` is returned
        instead of a plain ``MuScaledSOCProjection``.  The composite
        enforces both ``q = g(y)`` (algebraic) and ``r ∈ K_μ`` (SOC
        contact) through the semismooth Newton natural-map
        formulation.

        The ``q_slice`` ranges must **not** overlap the SOC reaction
        DOFs (which occupy indices ``n_phys`` to ``n_phys + n_react``
        in the augmented state).
    get_s0 : callable or None, optional
        Normal pre-stress offset for ``MuScaledSOCProjection``.
        Signature ``(y,)`` or ``(y, t)`` or ``(y, t, Fk_val)`` → scalar.
        Called on the full *augmented* state.
    get_w0 : callable or None, optional
        Tangential pre-stress offset.
        Signature ``(y, k)`` or ``(y, k, t)`` → ndarray matching the
        tangential dimension of block *k*.
    get_ds0_dz : callable or None, optional
        Jacobian of ``s0`` w.r.t. the state: ``(y,) → ndarray(n_aug)``.
    get_dw0_dz : callable or None, optional
        Jacobian of ``w0``: ``(y, k) → ndarray(m, n_aug)``.
    get_B : callable or None, optional
        Anisotropy matrix callback for elliptic friction cones.
        Signature ``get_B(y, k) -> ndarray(m, m)`` where *k* is the
        block index and *m* is the tangential dimension.  Must return
        a symmetric positive-definite matrix.  When provided, an
        ``AnisotropicSOCProjection`` is used instead of the default
        ``MuScaledSOCProjection``.  Direction-dependent friction can
        be encoded as ``B = diag(1/μ₁², 1/μ₂²)`` with ``mu=1``.
    step_size_ref : list or None, optional
        A mutable single-element list ``[h]`` whose first element is
        updated to the current step size on every RHS evaluation.
        Useful for ``get_s0`` / ``get_w0`` closures that need the
        adaptive step size (e.g. ``s_0 = F \cdot h``).
    rhs_jac : callable or None, optional
        Analytical Jacobian of ``rhs_smooth`` w.r.t. the *physical*
        state: ``rhs_jac(t, y_phys) -> ndarray(n_phys, n_phys)``.
        When provided, it is composed with the auto-generated coupling
        and De Saxcé blocks to form the full augmented Jacobian,
        returned as ``ContactSystem.rhs_jac``.

        When *None* (default), the augmented Jacobian uses column-wise
        finite differences on ``rhs_smooth`` *only* (not on the full
        augmented system).  This is significantly cheaper than
        differencing the augmented system since ``n_phys < n_aug``.
    Returns
    -------
    ContactSystem
        Dataclass whose attributes plug directly into
        :func:`solve_ivp_ns`.

    Notes
    -----
    The augmented RHS has signature ``rhs(t, y, prev_state, h)``
    compatible with ``BackwardEuler`` when ``pass_prev_state=True``
    and ``pass_step_size=True``.

    * **Physical rows:** ``f_smooth(t, x) + B @ p / (coupling_theta * h)``
    * **Reaction rows:** ``-û`` (De Saxcé–Frémond dual), always supplied
      regardless of gap.  The gap logic is handled by the projector via
      ``zero_inactive=True``.

    The Frémond augmented velocity is:

    .. math::
        \hat u_N = v_N + \mu\|v_T\| + c\, v_N^{\text{prev}}, \qquad
        \hat u_T = v_T

    where :math:`c = \theta(1+e)-1`.  For BE (:math:`\theta=1`),
    :math:`c = e` recovers the standard Moreau restitution.

    **Reaction coupling pre-compensation.**  The Moreau momentum
    balance is :math:`M(v_{n+1}-v_n) = h\,F_\theta + p_{n+1}`.
    The reaction impulse :math:`p` enters at **full** strength — it
    must NOT be scaled by the integrator coefficient.

    *Standard mode* (``incremental_coupling=False``): the physical rows
    pre-divide by ``coupling_theta``:
    ``B @ r / (coupling_theta * h)``.  After the integrator applies
    its own scaling, the net coupling is ``B r / h``.  Correct for
    Backward Euler and each SDIRK2 stage individually.

    *Anti-blend mode* (``incremental_coupling=True``): the physical rows
    use ``B (r_prev + (r - r_prev)/coupling_theta) / h``.
    When the integrator θ-blends old and new evaluations, the old
    term vanishes and the net coupling is exactly ``B r_new / h``.
    This eliminates ghost impulses from old-step reactions that would
    otherwise contaminate the momentum during contact→free transitions.

    *Frémond θ-averaged mode* (``fremond_contact=True``): the reaction
    rows evaluate the De Saxcé augmentation at :math:`v_{k+\theta}`
    (norm of the θ-blend, not θ-blend of norms).  An anti-blend
    formula ensures the integrator's θ-averaging yields exactly
    :math:`-\Theta(v_{k+\theta})` for the effective contact law.
    Positive dissipation is guaranteed for
    :math:`\tfrac{1}{2} \le \theta \le 1/(1+\bar e)`.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size

    # Default coupling_theta to theta (correct for one-step θ-methods)
    if coupling_theta is None:
        coupling_theta = theta

    # ── Validate C_extract / rate_form ───────────────────────────────
    if C_extract is not None:
        if sp.issparse(C_extract):
            C_extract = C_extract.tocsr()
        else:
            C_extract = np.asarray(C_extract, dtype=float)
        if D_extract is None:
            D_extract = C_extract
        elif sp.issparse(D_extract):
            D_extract = D_extract.tocsr()
        else:
            D_extract = np.asarray(D_extract, dtype=float)
        if C_extract.shape[1] != n_phys:
            raise ValueError(
                f"C_extract has {C_extract.shape[1]} columns but "
                f"n_phys = {n_phys}")
        if rate_form:
            for c in contacts:
                e_val = float(c.get('e', 0.0))
                c_check = theta * (1.0 + e_val) - 1.0
                if abs(c_check) > 1e-12:
                    raise ValueError(
                        f"rate_form=True requires θ(1+e)=1 (c_coeff=0) "
                        f"for each contact to avoid needing a previous-"
                        f"step velocity.  Got θ={theta}, e={e_val}, "
                        f"c={c_check:.6f}.  Use θ=1.0 with e=0.0.")
        if incremental_coupling or fremond_contact:
            raise NotImplementedError(
                "C_extract is not yet compatible with "
                "incremental_coupling or fremond_contact.  Use "
                "theta=1.0 (backward Euler) with standard coupling.")

    if gap_func is None and C_extract is None:
        raise ValueError(
            "gap_func must be provided when C_extract is None")

    # ── Frémond dissipation condition (Proposition 1) ────────────────
    if fremond_contact:
        e_max = max(float(c.get('e', 0.0)) for c in contacts)
        theta_max = 1.0 / (1.0 + e_max) if e_max > 0 else 1.0
        if theta < 0.5 - 1e-12 or theta > theta_max + 1e-12:
            raise ValueError(
                f"Frémond dissipation condition violated: θ must be in "
                f"[1/2, 1/(1+ē)] = [0.5, {theta_max:.6f}] "
                f"for ē = max(e) = {e_max:.4f}, got θ = {theta}"
            )

    # ── Normalise per-contact specs ──────────────────────────────────
    _contacts = []
    r_offset = n_phys
    reaction_idx = 0
    soc_blocks = []

    for k, c in enumerate(contacts):
        vN = int(c['vel_normal_idx'])
        vT = list(np.atleast_1d(c['vel_tangential_idx']).astype(int))
        mu_val = c.get('mu', 0.0)
        e_val = float(c.get('e', 0.0))
        # Frémond θ-averaged mode and standard mode both use
        # c = θ(1+e)−1.  Only the old incremental_coupling path
        # (fully-implicit algebraic rows) uses c = e.
        if incremental_coupling and not fremond_contact:
            c_coeff = e_val
        else:
            c_coeff = theta * (1.0 + e_val) - 1.0

        if callable(mu_val):
            _get_mu = mu_val
            _mu_is_const = False
        else:
            _m = float(mu_val)
            _get_mu = lambda y, _m=_m: _m   # noqa: E731
            _mu_is_const = True

        # Dilatancy coefficient: beta (default 0 = no dilatancy)
        beta_val = c.get('beta', 0.0)
        if callable(beta_val):
            _get_beta = beta_val
            _beta_is_const = False
        else:
            _b = float(beta_val)
            _get_beta = lambda y, _b=_b: _b  # noqa: E731
            _beta_is_const = True

        n_tang = len(vT)
        rN = r_offset + reaction_idx
        rT = [r_offset + reaction_idx + 1 + j for j in range(n_tang)]

        _contacts.append({
            'vN': vN, 'vT': vT,
            'rN': rN, 'rT': rT,
            'get_mu': _get_mu,
            'get_beta': _get_beta,
            'c_coeff': c_coeff,
            'mu_is_const': _mu_is_const,
            'beta_is_const': _beta_is_const,
        })
        soc_blocks.append((rN, rT))
        reaction_idx += 1 + n_tang

    n_react = reaction_idx
    n_aug = n_phys + n_react

    # ── Auto-generate B if not provided ──────────────────────────────
    if B is None and C_extract is not None:
        # Virtual-work pairing: B = C_extract^T (reordered to match
        # the reaction DOF ordering from the contacts list).
        reaction_extract_rows = []
        for ci in _contacts:
            reaction_extract_rows.append(ci['vN'])
            reaction_extract_rows.extend(ci['vT'])
        if sp.issparse(C_extract):
            B_mat = C_extract[reaction_extract_rows, :].T.tocsr()
        else:
            B_mat = C_extract[reaction_extract_rows, :].T.copy()
    elif B is None:
        # Identity coupling: each reaction acts on exactly one
        # physical DOF (valid for point-mass / direct-index problems).
        B_mat = np.zeros((n_phys, n_react))
        col = 0
        for ci in _contacts:
            B_mat[ci['vN'], col] = 1.0   # r_N → vel_normal_idx
            col += 1
            for vt in ci['vT']:
                B_mat[vt, col] = 1.0     # r_T_j → vel_tangential_idx_j
                col += 1
    else:
        if sp.issparse(B):
            B_mat = B.tocsr()
        else:
            B_mat = np.asarray(B)
        if B_mat.shape != (n_phys, n_react):
            raise ValueError(
                f"B shape {B_mat.shape} doesn't match "
                f"(n_phys={n_phys}, n_react={n_react})")

    # ── Augmented descriptor matrix ──────────────────────────────────
    if sp.issparse(A):
        A_aug = sp.block_diag(
            [A, sp.csr_matrix((n_react, n_react))], format='csr')
    else:
        A_aug = np.zeros((n_aug, n_aug))
        A_aug[:n_phys, :n_phys] = np.asarray(A)

    # ── Augmented initial condition ──────────────────────────────────
    y0_aug = np.zeros(n_aug)
    y0_aug[:n_phys] = y0

    # ── Gap wrapper (physical → augmented) ───────────────────────────
    _n = n_phys

    if gap_func is not None:
        _gap = gap_func
        def gap_aug(y, t):
            return _gap(y[:_n], t)
    else:
        # Auto-generate from C_extract normal rows.
        _C_gap = C_extract
        _normal_rows = [ci['vN'] for ci in _contacts]
        def gap_aug(y, t):
            contact_disp = np.asarray(_C_gap @ y[:_n]).ravel()
            return contact_disp[_normal_rows]

    # ── get_mu on augmented state ────────────────────────────────────
    _ci = _contacts

    def get_mu_aug(y):
        yp = y[:_n]
        return np.array([ci['get_mu'](yp) for ci in _ci])

    # ── Projection ───────────────────────────────────────────────────
    _soc_kw = dict(
        blocks=soc_blocks,
        get_mu=get_mu_aug,
        gap_func=gap_aug,
        zero_inactive=True,
    )
    if get_s0 is not None:
        _soc_kw['get_s0'] = get_s0
    if get_w0 is not None:
        _soc_kw['get_w0'] = get_w0
    if get_ds0_dz is not None:
        _soc_kw['get_ds0_dz'] = get_ds0_dz
    if get_dw0_dz is not None:
        _soc_kw['get_dw0_dz'] = get_dw0_dz

    if get_B is not None:
        _soc_kw['get_B'] = get_B
        soc_proj = AnisotropicSOCProjection(**_soc_kw)
    else:
        soc_proj = MuScaledSOCProjection(**_soc_kw)

    if constraints is not None:
        alg_proj = AlgebraicConstraintProjection(constraints=constraints)
        proj = CompositeContactProjection(alg_proj, soc_proj)
    else:
        proj = soc_proj

    # ── Velocity DOF map for active-set filtering ────────────────────
    # Maps each SOC block index to the velocity DOF indices in the
    # augmented state that are dynamically coupled to that contact.
    # Used by AdaptiveStepping to suppress error-norm contributions
    # from DOFs undergoing a stick↔slip or contact↔separation transition.
    _vel_dof_map = []
    for ci in _contacts:
        dofs = [ci['vN']] + list(ci['vT'])
        _vel_dof_map.append(np.array(dofs, dtype=int))
    # Attach to the SOC projection (CompositeContactProjection delegates)
    soc_proj._velocity_dof_map = _vel_dof_map

    # ── Augmented RHS ────────────────────────────────────────────────
    _B = B_mat
    _rhs = rhs_smooth
    _theta = float(coupling_theta)  # capture for reaction pre-compensation
    _fremond = bool(fremond_contact)  # Frémond θ-averaged contact mode
    _incr = bool(incremental_coupling and not fremond_contact)  # old anti-blend
    _incr_coupling = bool(incremental_coupling or fremond_contact)  # physical coupling
    _theta_c = float(theta)  # θ for Frémond reaction-row averaging
    _C = C_extract        # None → identity (direct indexing)
    _D_op = D_extract if D_extract is not None else C_extract
    _rate = bool(rate_form)
    _h_ref = step_size_ref  # mutable [h] updated on every RHS call

    # ── Vectorized precomputation for De Saxcé loops ─────────────────
    # When all contacts share the same tangential dimension and mu/beta
    # are constants (not callable), we can batch the reaction rows
    # into numpy operations instead of Python loops.
    _tang_dims = [len(ci['vT']) for ci in _contacts]
    _uniform_tang = (len(_tang_dims) > 0
                     and all(d_k == _tang_dims[0] for d_k in _tang_dims))
    _m_tang = _tang_dims[0] if _uniform_tang else 0

    if _uniform_tang and _C is None:
        _N_c = len(_contacts)
        _vN_idx = np.array([ci['vN'] for ci in _contacts], dtype=int)
        _vT_idx = np.array([ci['vT'] for ci in _contacts], dtype=int)  # (N_c, m)
        _rN_idx = np.array([ci['rN'] for ci in _contacts], dtype=int)
        _rT_idx = np.array([ci['rT'] for ci in _contacts], dtype=int)  # (N_c, m)
        # Local reaction indices (0-based, for the J_ds sub-block)
        _rN_loc = _rN_idx - n_phys
        _rT_loc = _rT_idx - n_phys
        _c_arr  = np.array([ci['c_coeff'] for ci in _contacts])

        # Check if mu/beta are all numeric constants (not state-dependent)
        _all_const = all(ci['mu_is_const'] and ci['beta_is_const']
                         for ci in _contacts)
        if _all_const:
            _mu_vals = np.empty(_N_c)
            _beta_vals = np.empty(_N_c)
            _dummy = np.zeros(n_phys)
            for k_i, ci in enumerate(_contacts):
                _mu_vals[k_i] = ci['get_mu'](_dummy)
                _beta_vals[k_i] = ci['get_beta'](_dummy)
            _alpha_arr = _mu_vals - _beta_vals
        else:
            _alpha_arr = None  # recomputed at each call
        _can_batch = True
        # Sparse B / (θ * 1.0) pattern (precomputed, scaled by h at runtime)
        if sp.issparse(_B):
            _B_sp = _B.tocsr()
        else:
            _B_sp = sp.csr_matrix(_B)
    else:
        _can_batch = False

    def rhs_aug(t, y, *extra):
        r"""Auto-generated augmented RHS.

        Physical rows:  smooth forces + impulse coupling.
        Reaction rows:  ``-û`` (De Saxcé–Frémond dual).

        Uses ``*extra`` to robustly accept whatever argument order
        ``_get_bound_wrapper`` resolves.
        """
        # ── Robustly extract prev_state and step size from *extra ──
        h_val = None
        prev_state = None
        for a in reversed(extra):
            if a is not None and np.isscalar(a):
                h_val = float(a)
                break
        for a in extra:
            if isinstance(a, np.ndarray) and a.shape == y.shape:
                prev_state = a
                break

        # Keep the mutable step-size reference up to date (if provided)
        if _h_ref is not None and h_val is not None and h_val > 0:
            _h_ref[0] = h_val

        out = np.zeros(len(y))

        # Physical rows: smooth forces + impulse coupling
        out[:_n] = _rhs(t, y[:_n])
        r = y[_n:]
        if h_val is not None and h_val > 0:
            if _incr_coupling and prev_state is not None:
                r_prev_r = prev_state[_n:]
                out[:_n] += _B @ (r_prev_r + (r - r_prev_r) / _theta) / h_val
            else:
                out[:_n] += _B @ r / (_theta * h_val)

        # ── Reaction rows: De Saxcé dual ──
        yp = y[:_n]

        # ── Vectorized fast-path (identity extraction, uniform m, standard mode) ──
        if _can_batch and not _fremond and not _incr:
            v_N = yp[_vN_idx]                       # (N_c,)
            v_T = yp[_vT_idx]                       # (N_c, m)
            if _m_tang == 1:
                v_T_norm = np.abs(v_T[:, 0])
            else:
                v_T_norm = np.linalg.norm(v_T, axis=1)

            v_N_prev = np.zeros(_N_c)
            if prev_state is not None:
                v_N_prev = prev_state[:_n][_vN_idx]

            # Re-evaluate alpha when mu/beta are state-dependent
            alpha = _alpha_arr
            if alpha is None:
                alpha = np.array([ci['get_mu'](yp) - ci['get_beta'](yp)
                                  for ci in _ci])

            out[_rN_idx] = -(v_N + alpha * v_T_norm + _c_arr * v_N_prev)
            out[_rT_idx] = -v_T
            return out

        # ── Scalar fallback (C_extract, Frémond, incremental, or non-uniform) ──
        if _C is not None:
            if _rate:
                if prev_state is not None and h_val is not None and h_val > 0:
                    _v_c = np.asarray(
                        _D_op @ (yp - prev_state[:_n]) / h_val).ravel()
                else:
                    _v_c = np.zeros(_D_op.shape[0])
                _v_c_prev = None
            else:
                _v_c = np.asarray(_D_op @ yp).ravel()
                _v_c_prev = (np.asarray(_D_op @ prev_state[:_n]).ravel()
                             if prev_state is not None else None)
        else:
            _v_c = yp
            _v_c_prev = prev_state[:_n] if prev_state is not None else None

        for ci in _ci:
            v_N = float(_v_c[ci['vN']])
            vT_idx = ci['vT']
            v_T = _v_c[np.asarray(vT_idx)]
            v_T_norm = float(np.linalg.norm(v_T))
            mu_k = ci['get_mu'](yp)
            beta_k = ci['get_beta'](yp)
            alpha_k = mu_k - beta_k
            c_k = ci['c_coeff']

            v_N_prev = 0.0
            if _v_c_prev is not None:
                v_N_prev = float(_v_c_prev[ci['vN']])

            uhat_N = v_N + alpha_k * v_T_norm + c_k * v_N_prev
            uhat_T = v_T.copy()

            if _fremond and prev_state is not None:
                yp_prev = prev_state[:_n]
                v_N_old = float(yp_prev[ci['vN']])
                v_T_old = yp_prev[np.asarray(vT_idx)]
                v_N_th = _theta_c * v_N + (1.0 - _theta_c) * v_N_old
                v_T_th = _theta_c * v_T + (1.0 - _theta_c) * v_T_old
                v_T_th_norm = float(np.linalg.norm(v_T_th))
                Theta_N = v_N_th + c_k * v_N_prev + alpha_k * v_T_th_norm
                Theta_T = v_T_th.copy()
                v_T_old_norm = float(np.linalg.norm(v_T_old))
                mu_old = ci['get_mu'](yp_prev)
                beta_old = ci['get_beta'](yp_prev)
                alpha_old = mu_old - beta_old
                Theta_N_old = (1.0 + c_k) * v_N_old + alpha_old * v_T_old_norm
                Theta_T_old = v_T_old.copy()
                out[ci['rN']] = (-Theta_N + (1.0 - _theta_c) * Theta_N_old) / _theta_c
                for j, t_idx in enumerate(ci['rT']):
                    out[t_idx] = (-Theta_T[j] + (1.0 - _theta_c) * Theta_T_old[j]) / _theta_c
            elif _incr and prev_state is not None:
                yp_prev = prev_state[:_n]
                v_N_prev_eval = float(yp_prev[ci['vN']])
                v_T_prev_eval = yp_prev[np.asarray(vT_idx)]
                v_T_prev_norm = float(np.linalg.norm(v_T_prev_eval))
                mu_prev = ci['get_mu'](yp_prev)
                beta_prev = ci['get_beta'](yp_prev)
                alpha_prev = mu_prev - beta_prev
                uhat_N_prev = v_N_prev_eval + alpha_prev * v_T_prev_norm + c_k * v_N_prev
                uhat_T_prev = v_T_prev_eval.copy()
                out[ci['rN']] = -(uhat_N - uhat_N_prev) / _theta - uhat_N_prev
                for j, t_idx in enumerate(ci['rT']):
                    out[t_idx] = -(uhat_T[j] - uhat_T_prev[j]) / _theta - uhat_T_prev[j]
            else:
                out[ci['rN']] = -uhat_N
                for j, t_idx in enumerate(ci['rT']):
                    out[t_idx] = -uhat_T[j]

        return out

    # ── Component slices ─────────────────────────────────────────────
    # Accept slice objects, index arrays (list/ndarray of int), or None.
    # When None, auto-generate from the contact specification:
    #   block 0: velocity DOFs (from vel_normal_idx / vel_tangential_idx)
    #   block 1: remaining physical DOFs (positions / other)
    #   block 2: reaction DOFs
    # This works even when normal and tangential DOFs are non-contiguous.
    if component_slices is not None:
        _cs_norm = []
        for cs_item in component_slices:
            if isinstance(cs_item, slice):
                _cs_norm.append(cs_item)
            else:
                _cs_norm.append(np.asarray(cs_item, dtype=int))
        # Append reaction DOFs in the same style as the user's entries:
        # use an index array when any entry is an array, else a slice.
        _any_array = any(isinstance(c, np.ndarray) for c in _cs_norm)
        if _any_array:
            _cs_norm.append(np.arange(n_phys, n_aug, dtype=int))
        else:
            _cs_norm.append(slice(n_phys, n_aug))
        cs_aug = _cs_norm
    else:
        # Auto-generate: gather velocity DOFs from contacts, rest are
        # positions / other.  Handles non-contiguous DOF layouts.
        _vel_set = set()
        for ci in _contacts:
            _vel_set.add(ci['vN'])
            _vel_set.update(ci['vT'])
        _vel_idx = np.array(sorted(_vel_set), dtype=int)
        _pos_idx = np.array(sorted(set(range(n_phys)) - _vel_set), dtype=int)
        _react_idx = np.arange(n_phys, n_aug, dtype=int)
        cs_aug = []
        if _vel_idx.size > 0:
            cs_aug.append(_vel_idx)
        if _pos_idx.size > 0:
            cs_aug.append(_pos_idx)
        cs_aug.append(_react_idx)

    integrator_opts = {
        'pass_prev_state': True,
        'pass_step_size': True,
    }

    # ── Auto-generated augmented Jacobian ────────────────────────────
    _user_rhs_jac = rhs_jac
    _n_aug_jac = n_aug

    def _fd_smooth_jac(t, y_phys):
        """Column-wise FD Jacobian of rhs_smooth (n_phys DOFs only)."""
        f0 = _rhs(t, y_phys)
        n_p = len(y_phys)
        eps_base = 1e-7
        h_vec = eps_base * np.maximum(np.abs(y_phys), 1.0)
        J = np.empty((n_p, n_p))
        for i in range(n_p):
            y_pert = y_phys.copy()
            y_pert[i] += h_vec[i]
            J[:, i] = (_rhs(t, y_pert) - f0) / h_vec[i]
        return J

    def jac_aug(t, y, *extra):
        r"""Auto-generated Jacobian of the augmented RHS.

        Returns sparse CSR when ``_can_batch`` is True (identity
        extraction, uniform tangential dimension), otherwise dense.

        Structure::

            J = [ J_smooth        B/(θ h)  ]   (physical rows)
                [ J_desaxce       0        ]   (reaction rows)
        """
        # ── Parse extra args ──
        h_val = None
        prev_state = None
        for a in reversed(extra):
            if a is not None and np.isscalar(a):
                h_val = float(a)
                break
        for a in extra:
            if isinstance(a, np.ndarray) and a.shape == y.shape:
                prev_state = a
                break
        if _h_ref is not None:
            if h_val is None or h_val <= 0:
                h_val = _h_ref[0]
        if h_val is None or h_val <= 0:
            h_val = 1.0

        yp = y[:_n]

        # ── Sparse fast-path (identity extraction, uniform m, standard mode) ──
        if _can_batch and not _fremond and not _incr:
            # 1) Smooth RHS Jacobian
            if _user_rhs_jac is not None:
                J_s = _user_rhs_jac(t, yp)
            else:
                J_s = _fd_smooth_jac(t, yp)
            if not sp.issparse(J_s):
                J_s = sp.csr_matrix(J_s)

            # 2) Coupling: B / (θ h) — sparse
            B_coup = _B_sp / (_theta * h_val)

            # 3) De Saxcé derivatives (vectorized COO)
            v_T = yp[_vT_idx]                     # (N_c, m)
            if _m_tang == 1:
                v_T_norm = np.abs(v_T[:, 0])
            else:
                v_T_norm = np.linalg.norm(v_T, axis=1)

            # Rows: rN→vN, rN→vT_j (if ||vT||>eps), rT_j→vT_j
            # For m=1: up to 3 entries per contact
            ds_rows = []
            ds_cols = []
            ds_data = []

            # rN → vN:  ∂(-û_N)/∂v_N = -1
            ds_rows.append(_rN_loc)
            ds_cols.append(_vN_idx)
            ds_data.append(np.full(_N_c, -1.0))

            # rN → vT_j:  ∂(-û_N)/∂v_T_j = -α * v_T_j / ||v_T||
            # Re-evaluate alpha when mu/beta are state-dependent
            alpha = _alpha_arr
            if alpha is None:
                alpha = np.array([ci['get_mu'](yp) - ci['get_beta'](yp)
                                  for ci in _ci])
            for j in range(_m_tang):
                nz = v_T_norm > 1e-14
                if np.any(nz):
                    ds_rows.append(_rN_loc[nz])
                    ds_cols.append(_vT_idx[nz, j])
                    ds_data.append(-alpha[nz] * v_T[:, j][nz] / v_T_norm[nz])

            # rT_j → vT_j:  ∂(-û_T_j)/∂v_T_j = -1
            for j in range(_m_tang):
                ds_rows.append(_rT_loc[:, j])
                ds_cols.append(_vT_idx[:, j])
                ds_data.append(np.full(_N_c, -1.0))

            J_ds = sp.csr_matrix(
                (np.concatenate(ds_data),
                 (np.concatenate(ds_rows), np.concatenate(ds_cols))),
                shape=(n_react, _n))

            # 4) Assemble block matrix
            return sp.bmat([
                [J_s,   B_coup],
                [J_ds,  None  ],
            ], format='csr')

        # ── Dense fallback ──
        J = np.zeros((_n_aug_jac, _n_aug_jac))

        if _user_rhs_jac is not None:
            J_s = _user_rhs_jac(t, yp)
            if sp.issparse(J_s):
                J[:_n, :_n] = J_s.toarray()
            else:
                J[:_n, :_n] = J_s
        else:
            J[:_n, :_n] = _fd_smooth_jac(t, yp)

        if sp.issparse(_B):
            J[:_n, _n:] = (_B / (_theta * h_val)).toarray()
        else:
            J[:_n, _n:] = _B / (_theta * h_val)

        if _C is not None:
            if _rate:
                if prev_state is not None and h_val > 0:
                    v_c = np.asarray(
                        _D_op @ (yp - prev_state[:_n]) / h_val).ravel()
                    dvc_dyp = ((_D_op / h_val).toarray() if sp.issparse(_D_op)
                               else _D_op / h_val)
                else:
                    v_c = np.zeros(_D_op.shape[0])
                    dvc_dyp = np.zeros((_D_op.shape[0], _n))
            else:
                v_c = np.asarray(_D_op @ yp).ravel()
                dvc_dyp = (_D_op.toarray() if sp.issparse(_D_op)
                           else np.asarray(_D_op))
        else:
            v_c = yp
            dvc_dyp = None

        for ci_k in _ci:
            vN = ci_k['vN']
            vT = ci_k['vT']
            rN = ci_k['rN']
            rT = ci_k['rT']
            mu_k = ci_k['get_mu'](yp)
            beta_k = ci_k['get_beta'](yp)
            alpha_k = mu_k - beta_k
            v_T = v_c[np.asarray(vT)]

            if _fremond and prev_state is not None:
                if _C is not None:
                    v_c_prev = np.asarray(_D_op @ prev_state[:_n]).ravel()
                else:
                    v_c_prev = prev_state[:_n]
                v_T_old = v_c_prev[np.asarray(vT)]
                v_T_dir = _theta_c * v_T + (1.0 - _theta_c) * v_T_old
                react_scale = 1.0
            elif _incr and prev_state is not None:
                v_T_dir = v_T
                react_scale = 1.0 / _theta
            else:
                v_T_dir = v_T
                react_scale = 1.0

            v_T_dir_norm = float(np.linalg.norm(v_T_dir))

            if dvc_dyp is not None:
                J[rN, :_n] = -react_scale * dvc_dyp[vN, :]
                if v_T_dir_norm > 1e-14:
                    for j, vt_j in enumerate(vT):
                        J[rN, :_n] -= (react_scale * alpha_k
                                       * v_T_dir[j] / v_T_dir_norm
                                       * dvc_dyp[vt_j, :])
                for j_idx, (t_row, vt_j) in enumerate(zip(rT, vT)):
                    J[t_row, :_n] = -react_scale * dvc_dyp[vt_j, :]
            else:
                J[rN, vN] = -react_scale
                if v_T_dir_norm > 1e-14:
                    for j, vt_j in enumerate(vT):
                        J[rN, vt_j] = (-react_scale * alpha_k
                                       * v_T_dir[j] / v_T_dir_norm)
                for t_row, vt_j in zip(rT, vT):
                    J[t_row, vt_j] = -react_scale

        return J

    return ContactSystem(
        A=A_aug,
        rhs=rhs_aug,
        y0=y0_aug,
        projection=proj,
        component_slices=cs_aug,
        integrator_opts=integrator_opts,
        n_phys=n_phys,
        B=B_mat,
        rhs_jac=jac_aug,
    )
