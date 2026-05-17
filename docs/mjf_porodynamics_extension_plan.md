# Plan — extending MJF to general porodynamics with crack pressure

Scope of this document: what's needed to drive
`solve_nivp.moreau_jean_fremond` from the existing embedded-crack
poroelasticity setup, and what extensions are needed to support pressure,
fluid velocity, and crack pressure as additional state components.

## Where the existing MJF stepper stops short

The current `MoreauJeanFremondStepper` accepts a clean block decomposition:

```
state = (q, v, p_pore)
[[ M̂            θ B_biot^T ]] [v_kt]   [ rhs_v + θ H^T p_contact ]
[[ θ B_biot     Ŝ           ]] [p_kt] = [ rhs_p                    ]
```

The embedded-crack notebook gives us instead:

- A single descriptor matrix `A_dyn_s` (built by
  `poro_s.build_first_order_dynamic_system`) with mixed pressure,
  velocity, displacement, and algebraic-multiplier rows.
- A combined `rhs_dyn_s_eff` that folds in pressure-normal coupling,
  Lysmer absorbing damping, and Winkler stiffness.
- Algebraic DAE constraints (`flux_constraint_s`,
  `zero_avg_lam_s_s`) enforced via Lagrange multipliers in the
  augmented state.
- Reaction-state mirroring (`reaction_state_indices_s`) that stores the
  contact reaction inside the augmented `y` vector.
- Prestress offsets (`get_s0`, `get_w0`) that shift the contact reaction
  by a quasi-static initial stress state.

None of these features map directly onto the current MJF interface.
The starter cells in `examples/embedded_crack_mc_sliding_mjf.ipynb`
attempt to extract `(M, K, C)` for the elasticity-only sub-block from
`A_dyn_s` and `rhs_jac_dyn_s_eff` and ignore the rest; that's a
demonstration only, not a faithful conversion.

## What a faithful conversion needs

### A. State-vector decomposition

Generalize the stepper's state from `(q, v, p_pore)` to a flat
descriptor list. Concrete target:

```
state = (q, v, p_pore, w_fluid, p_crack, λ)
```

with

| Component | Symbol | Block size | Continuous-time eqn |
|---|---|---|---|
| Solid displacement | `q` | `n_solid` | `q̇ = v` |
| Solid velocity | `v` | `n_solid` | `M v̇ + K q + C v + Bp_pore^T p_pore + Bpc^T p_crack = F + H^T p_contact` |
| Pore pressure | `p_pore` | `n_pore` | `S ṗ_pore + D_pp p_pore + B_pv v + B_pw w_fluid = s_p` |
| Darcy / Brinkman fluid velocity | `w_fluid` | `n_fluid_v` | `M_f ẇ + K_w w + C_w v + B_wp p_pore = F_f` |
| Crack-resident pressure | `p_crack` | `n_crack` | `S_c ṗ_crack + D_c p_crack + B_cv·u_normal + B_cw·u_T = s_c + (contact-driven mass exchange)` |
| Algebraic multipliers | `λ` | `n_alg` | `g(state) = 0` (e.g. flux balance, mean traction zero) |

Not every problem will have all components. The interface should let
the user declare which components are present and supply only the
relevant block matrices.

### B. Augmented operator assembly

Generalize `_build_aug_operator` to accept a list of
`(state_component, mass_block, stiffness_block, damping_block)` tuples
plus a coupling matrix that fills the off-diagonal blocks. Concrete
shape, with `n_components = 4` say:

```
Aug = [[ M̂_qq      M̂_qp_pore     M̂_qw      M̂_qpc ]
       [ M̂_p_poreq M̂_p_porep_pore M̂_p_porew M̂_p_porepc ]
       [ M̂_wq      M̂_wp_pore     M̂_ww      M̂_wpc ]
       [ M̂_pcq     M̂_pcp_pore    M̂_pcw     M̂_pcpc ]]
```

where each `M̂_xy = (1-θ) M_xy + θ K_xy h^2 + (hθ) C_xy` for diagonal
blocks and the off-diagonal Biot-style couplings get their own
`(hθ)` scaling. The contact reaction `H^T p_contact` enters only on the
solid-velocity block; if there is *also* contact-driven fluid mass
exchange (crack pressure responds to slip) it would enter on the
`p_crack` block via a different operator `H_crack^T p_contact`.

The current dense-or-sparse `_block2x2` helper needs to become a
`_block_NxN` helper. Either build the augmented operator as a single
sparse `bmat` once at construction (constant blocks) or as a function
`build_aug_operator(h)` that re-assembles per step.

For descriptor-form input (a single `A_descriptor` matrix as in the
embedded-crack notebook), provide an alternative path:

```python
build_moreau_jean_fremond_descriptor(
    A_descriptor, K_descriptor, C_descriptor,
    state_layout=...,   # how to slice (q, v, p_pore, w, pc, λ)
    H_descriptor=...,   # contact rows of the velocity block
    ...
)
```

Internally this just sets `M̂_total = A_descriptor + h·θ·C_descriptor +
(h·θ)²·K_descriptor` and treats it as one big mass-equivalent matrix.
Per step solve `M̂_total @ z_kt = rhs - θ H^T p_contact`. This avoids
extracting blocks at all but loses the energy-bound proof (which
relies on the block decomposition).

### C. Contact reaction and Delassus matrix in extended state

Generalize the Delassus assembly. With the multi-component state:

- Solve `Aug @ X = E_solid_v · H_dense.T` where `E_solid_v` is the
  embedding from solid-velocity rows into the full augmented row index.
- Contact velocity at the predictor: `u_pred = H @ E_solid_v_T @ z_pred`.
- Delassus: `W = θ · H @ E_solid_v_T @ X`.

If contact also drives crack pressure (e.g. fault-mass exchange),
extend to:

- Contact reaction enters at solid-velocity *and* crack-pressure rows
  via two H matrices: `H_solid` and `H_crack`.
- Define an extended reaction `p_contact_ext = (p_normal, p_tangential,
  p_fluid_exchange)` of size `2·n_contacts + n_crack` per contact (or
  similar).
- Build a generalized Delassus that projects the augmented inverse onto
  *both* the solid-velocity and crack-pressure rows.
- The SOCCP cone becomes a product cone:
  `K_friction × R^{n_crack}` if the fluid exchange is sign-free, or
  `K_friction × R_+^{n_crack}` if there's a unilateral mass-balance
  (e.g. fluid only flows out of the crack into the bulk when crack
  pressure exceeds bulk pressure).

### D. Algebraic constraints

The DAE multipliers `λ` need explicit handling. Two options:

**Option 1 — null-space projection at the predictor.** Project the
predictor onto the constraint manifold by solving an auxiliary KKT
system after the augmented predictor solve. Inexact constraint
satisfaction tolerated to round-off; same convergence rate as
descriptor Radau.

**Option 2 — augmented Lagrangian merge.** Add the constraint rows
into the augmented operator as `[...; G^T 0]` and `[...; G 0]` rows.
The contact reaction does not enter these constraint rows. The
augmented operator becomes singular and must be solved via a regularized
or null-space method.

Option 1 is cleaner; option 2 is closer to what the descriptor Radau
backend does today and reuses existing infrastructure.

### E. Prestress and reaction offsets

Today MJF treats the contact problem as `−Φ(W p + b) ∈ N_K(p)` with `p`
the *full* contact reaction (impulse). The embedded-crack notebook
splits the reaction into:

- A **prestress** part `(s0, w0)` that's known a priori (lithostat +
  far-field shear).
- A **perturbation** part computed by the SOCCP.

This split is just a change of variable: define `p = p_perturbation` and
let `b ← b + W·(s0, w0)` to absorb the prestress as a bias on the
predictor offset. The cone constraint becomes `p_pert + (s0,w0) ∈ K`,
which is a *shifted* cone test. Easy to add: accept `p_offset` in
`build_moreau_jean_fremond` and bias the SOCCP solve accordingly.

### F. Reaction-state mirror

The Radau backend stores the contact reaction inside the augmented
state vector via `reaction_state_indices`. MJF currently passes
`p_contact_prev` through `aux` for warm-start only. To support the
notebook's diagnostic and post-processing infrastructure unchanged, MJF
should optionally mirror the reaction into a designated slice of the
state vector at the end of each step. One-line addition during state
recovery.

### G. State-dependent friction with porothermal coupling

The current `aux_law='slip_weakening'` updates μ from cumulative slip
only. Real fault problems need μ to depend on *also*:

- Local pore pressure `p_crack[k]` (effective normal stress).
- Fluid velocity (advective heating).
- Crack aperture (gouge thickness).

Generalize to `mu_callable(state, aux, k)` — the user supplies a
function that takes the full augmented state and returns per-contact
μ. The state-derivative `∂μ/∂y` needs to be supplied (or finite-
differenced as today) for the SOCCP Jacobian.

For thermal pressurization, an additional aux variable per contact
(temperature) needs its own evolution equation. That's a generic
extension hook — same shape as the rate-state law.

### H. Adaptive time-stepping

The MJF stepper is fixed-step today. The embedded-crack notebook uses
adaptive Radau via `solve_nivp.solve_ivp_ns`. To match: wrap the MJF
stepper in an outer adaptive controller using:

- Richardson estimate (full step vs two half-steps).
- Or embedded-formula error if a higher-order θ-pair is added (e.g.
  Crank-Nicolson + backward-Euler comparison).

This is independent of the porodynamics extension and can be added
later as a `solve_nivp.AdaptiveMoreauJeanFremondStepper` wrapper.

## Concrete next-session task list

In rough order of payoff:

1. **Generalize `_build_aug_operator`** to N-component blocks with a
   builder API that takes a list of components and an optional coupling
   matrix.
2. **Add prestress / reaction-offset support** — small change, big
   payoff for matching the notebook's reaction conventions.
3. **Add reaction-state mirror** — preserves diagnostic infrastructure
   from the Radau backend.
4. **Add descriptor-form path**
   (`build_moreau_jean_fremond_descriptor`) — quickest way to consume
   the embedded-crack `A_dyn_s` without bothering with block extraction.
5. **Add algebraic-constraint handling** (null-space projection at the
   predictor) — required for the DAE constraints in the notebook.
6. **Generalize contact reaction to multiple H operators** for
   crack-pressure / fluid-mass coupling.
7. **State-dependent friction with porothermal arguments**.
8. **Adaptive time-stepping wrapper**.

Items 1–4 land the embedded-crack conversion. Items 5–6 are needed for
fault-injection problems with crack pressure. Items 7–8 are
infrastructure improvements that apply to all problems.

## Reference notebook artifact

`examples/embedded_crack_mc_sliding_mjf.ipynb` is the file-system copy
of the original sliding notebook with three new cells inserted after
the Radau solve (cells 15–17 in the new layout):

- A markdown explainer of conversion limitations.
- An assembly cell that extracts `(M, K, C)` for the elasticity-only
  sub-block and runs an MJF stepping loop with `e_N = 0`, `θ = 1/2`,
  `aux_law='constant'`.
- A comparison-plot cell.

These cells are **not** a working production conversion — they
demonstrate the extraction approach for the elasticity-only case and
will likely need debugging to actually run. Treat them as a starting
point. The full conversion requires the work items above.

## Prior art cross-reference

- Acary–Brogliato 2008, *Numerical Methods for Nonsmooth Dynamical
  Systems*, ch. 11 (Moreau-Jean) and ch. 12 (block PGS).
- Acary–Collins-Craft 2025 (HAL-04230941), *A second-order Moreau-Jean
  scheme with the Frémond impact law* — the source for the average-
  velocity formulation and Proposition 1 energy bound.
- Siconos source (Acary, Bremond, Huber et al.) — reference C++
  implementation of block PGS + local SOCCP solvers for production-
  scale problems; useful as a structural template if the descriptor
  path proves insufficient.
- Garagash–Germanovich, *Hydraulic fracture and dyke propagation* —
  reference physics for fault-injection runs that need the crack-
  pressure coupling.
