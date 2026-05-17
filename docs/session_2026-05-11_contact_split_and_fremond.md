# Session summary — 2026-05-11: Breuling product-cone split + θ-Moreau-Jean-Frémond integrator

## Motivation

Two related but independent threads:

1. The 2026-05-07 patch made `SOCFischerBurmeisterLaw` velocity-level at every
   internal Radau stage. This deviates from Breuling's projected scheme,
   which uses a **product-cone** at internal stages (position-level
   Signorini for the normal row + velocity-level Coulomb compliance for
   the tangential rows) and the velocity projection only at the endpoint.
   The deviation has no proven uniqueness theorem and can mask the Kane
   pathology in regimes with regime transitions. The fix is to make
   `'soc_fb'` route Stage 1 to NCP-FB (Breuling-correct product cone) and
   Stage 2 to the De Saxcé / SOC-FB bipotential where velocity-cone
   duality is variationally clean.
2. Even with the split, the endpoint Newton-Coulomb impact law inherits
   the Kane pathology when restitution `e_N > 0`. Acary-Collins-Craft
   2025 (HAL-04230941) fix this by enforcing the contact law on the
   θ-weighted average velocity `u_{k+θ} = (1−θ) u_k + θ u_{k+1}` rather
   than on `u_{k+1}` alone — at θ=½ this is the Frémond average-velocity
   formulation, which is unconditionally energy-dissipative across
   stick / slip / take-off / impact regardless of `e ∈ [0, 1]`, `μ ≥ 0`,
   and contact coupling.

We want a separate first/second-order θ-method integrator with the
Frémond shift, sized for general porodynamics and slip-rate friction
(slip-weakening, Dieterich-Ruina aging, Ruina slip), to use as an
energy-conservation reference and as a production scheme on impact-heavy
problems where Radau IIA's Newton-Coulomb endpoint is unsafe.

## Changes

### 1. `solve_nivp/projected_radau_contact.py` — stage / endpoint law split

`ProjectedRadauContactModel` now carries two laws:

- `contact_law` — used at internal Radau stages (the dispatch in
  `_contact_residual` / `_contact_jacobian` already routes by the law's
  `expects_velocity_normal` flag).
- `endpoint_law` — used by `_endpoint_contact_residual` /
  `_endpoint_contact_jacobian`.

When `endpoint_law` is left `None` it mirrors `contact_law`, preserving
backwards compatibility for every caller that did not explicitly pass an
endpoint law.

`build_projected_radau_contact` accepts a new keyword `endpoint_law=` and
the string dispatch for `contact_law` is updated:

| String | Stage 1 (`contact_law`) | Stage 2 (`endpoint_law`) | Notes |
|---|---|---|---|
| `'fischer_burmeister'` (default) | NCP-FB product cone | NCP-FB product cone (velocity-level at endpoint) | Pure Breuling. |
| `'soc_fb'` | NCP-FB product cone | `SOCFischerBurmeisterLaw` (De Saxcé bipotential) | Breuling Stage 1 + De Saxcé Stage 2. |
| `'soc_fb_uniform'` | `SOCFischerBurmeisterLaw` | `SOCFischerBurmeisterLaw` | Legacy single-law SOC-FB at both stages, kept for regression comparison. |
| `'desaxce'` | `DeSaxceProjectedConeLaw` | `DeSaxceProjectedConeLaw` | Unchanged. |

The semantic change is to `'soc_fb'`: previously this set a single
SOC-FB law at both stages; now it splits. Users who relied on the
legacy behavior should switch to `'soc_fb_uniform'`. The change is
documented inline in the builder.

`_law_residual_and_jac` gained a `law=` keyword argument that defaults
to `self.contact_law` for backwards compatibility. The endpoint Jacobian
path now passes `law=self.endpoint_law` through this entry point so the
μ-derivative finite-difference is computed against the correct law.

`_endpoint_contact_residual` and `_endpoint_contact_jacobian` now read
`self.endpoint_law` for the residual call. The endpoint dispatch is
uniformly velocity-level — `xi[0]` is the restitution-shifted normal
velocity, used both as the velocity-level Signorini argument for scalar
NCP laws and as the kinematic input for SOC-FB / De Saxcé.

### 2. `tests/test_projected_radau_contact.py` — single test migrated to legacy alias

`test_projected_radau_inplace_reaction_step_writes_existing_state`
deliberately uses a contrived setup where `D_extract` selects the
*velocity* and `C_extract` selects the *position* — so the "gap" the
NCP scalar dispatch sees at Stage 1 is the position (admissible at the
test state) while the *velocity* needs the impact reaction. Under the
new `'soc_fb'` semantics the position-level Stage 1 correctly does not
generate a contact reaction; the regression intent of this test is the
velocity-level Stage 1 dispatch from the legacy single-law SOC-FB.
Migrated to `'soc_fb_uniform'` with a comment explaining the choice.

The other two `'soc_fb'` tests (lines 47, 102) pass under both the new
and legacy semantics because their setups use `D_extract = C_extract` so
gap and contact-velocity coincide.

### 3. `solve_nivp/soccp_pgs.py` — block PGS solver for SOCCP (419 lines, new)

Standalone solver for `−Φ(W p + b) ∈ N_K(p)` where `K` is the product of
Coulomb cones over contact blocks. Reusable from any integrator.

- **Outer loop**: block-projected Gauss-Seidel with optional SOR
  (`sor_omega ∈ [1, 2)`).
- **Inner loop**: per-contact semismooth Newton on the SOC
  Fischer-Burmeister Jordan-algebra residual with Armijo line search,
  reusing `_soc_fb_phi_and_jac` from `projected_radau_contact.py`.
- **Shift hooks**: `desaxce_shift_factory(mu_vec)` and
  `fremond_shift_factory(mu_vec, e_N_vec, u_N_old, theta=...)` build
  per-contact shift closures. The local solver receives a generic
  `shift_fn(u_block, contact_index)` callback so any integrator can
  supply its own kinematic shift.
- **Diagnostics**: returns `SocppPgsInfo` with outer iterations, total
  inner iterations, convergence flag, residual, and per-contact regime
  classification (`separation` / `stick` / `slip`).

Per-step asymptotic cost is `O(K_outer · N² · d²)` for matrix-vector
with the Delassus matrix `W`, versus `O(N³ · d³)` for assembly +
factorization of a global SOCCP residual Jacobian. For `N ≫ 100`
contacts the PGS form is decisively faster; for `N < 50` either approach
is comparable.

### 4. `solve_nivp/moreau_jean_fremond.py` — θ+Frémond integrator (699 lines, new)

`MoreauJeanFremondStepper` advances `(q, v, p_pore, aux)` by one step,
solving a single block-augmented linear system for the predictor + a
SOCCP for the contact impulses, with the contact law evaluated on the
θ-weighted average velocity.

- **Block-augmented operator** for general porodynamics:

  ```
  Aug = [[ M̂,      hθ B_biot^T ],
         [ hθ B_biot,   Ŝ        ]]
  M̂ = M + hθC + (hθ)² K
  Ŝ = S + hθD
  ```

  Pure mechanics (`S = D = B_biot = None`) collapses to `Aug = M̂`.
  Contact reaction enters only on the solid block via `θ H^T p_contact`.

- **Delassus** `W = θ · H · (aug_inv solid block) · H^T` is built by
  solving `Aug @ X = [H^T; 0]` for the augmented system; the top
  `n_solid` rows of `X` are projected by `H`. Sparse and dense
  augmented operators are both supported.

- **Frémond shift** at θ=½: `Θ(u, k) = u + (μ‖u_T‖ + ½(e_N − 1)·u_N_old, 0)`,
  which collapses to De Saxcé at θ=1, e=0 and to the average-velocity
  form at θ=½ that gives Acary-Collins-Craft Proposition 1's
  unconditional energy bound.

- **θ-stability check** at construction: enforces
  `θ ∈ [½, 1/(1+e_max)]`. This is the Acary-Collins-Craft Prop. 1
  window; outside it the discrete energy decomposition is not
  guaranteed to be non-positive.

- **Auxiliary state laws** for slip-rate friction, swappable by name or
  callable:

  | `aux_law` | Update rule | μ formula |
  |---|---|---|
  | `'constant'` (default) | μ does not evolve | as supplied |
  | `'slip_weakening'` | `δ̇ = ‖u_T‖`, integrated explicitly | `μ_s − (μ_s − μ_d) min(δ/D_c, 1)` linear |
  | `'rate_state_aging'` (Dieterich) | `θ̇ = 1 − Vθ/L`, implicit | `μ_0 + a ln(V/V_0) + b ln(V_0θ/L)` |
  | `'rate_state_slip'` (Ruina) | `θ̇ = −(Vθ/L) ln(Vθ/L)`, explicit | same form as aging |
  | callable | user-supplied | user-supplied |

  The `'slip_weakening'` builder defaults pull `μ_s` from `mu_init` so
  passing `mu_init=0.4` does not get clobbered by the literal default
  `μ_s=0.6`.

- **Energy diagnostic** (`return_diagnostic=True`) returns
  `(dE_mech, W_ext, W_damp, W_contact, dE_theta, slack)` per step.
  `slack ≥ 0` is the Acary-Collins-Craft Prop. 1 dissipation budget;
  it should be non-negative to round-off across all regimes.

- **SOCCP warm-start** via `aux['p_contact_prev']` cuts subsequent-step
  iteration counts.

### 5. `tests/test_soccp_pgs.py` — 13 unit tests (174 lines, new)

Cover frictionless / sticking / sliding 2D and 3D contacts, two-contact
off-block coupling, separation, De Saxcé and Frémond shift identities,
warm-start, local-solver residual at solution, and input-non-mutation.

### 6. `tests/test_moreau_jean_fremond.py` — 13 integrator tests (346 lines, new)

Cover smooth oscillator (energy at θ=½ and damping at θ=½),
constant-force drift, inelastic impact (e=0), elastic impact (e=1),
partial restitution (e=½), `θ` admissibility check, 2D sliding-block
sticking and sliding regimes with Coulomb cone admissibility,
slip-weakening evolution, 1D Biot consolidation (porodynamics smooth
path), warm-start iteration reduction, and the energy-slack
non-negativity sweep across stick / slip / take-off regimes.

## Test results

```
$ pytest tests/ --tb=line -q --ignore=tests/test_prestressed_fault_dynamic_helper.py \
                                  --ignore=tests/test_rattle_local_slider.py
685 passed
```

The two ignored modules are pre-existing missing-helper imports
unrelated to this work; they were ignored before this session too.

Subset results:

```
$ pytest tests/test_projected_radau_contact.py -q          # 21 passed (1 migrated to soc_fb_uniform)
$ pytest tests/test_soccp_pgs.py -q                        # 13 passed
$ pytest tests/test_moreau_jean_fremond.py -q              # 13 passed
```

## Files changed

| File | Status | Lines |
|---|---|---|
| `solve_nivp/projected_radau_contact.py` | edited | +50 / −10 |
| `solve_nivp/soccp_pgs.py` | new | 419 |
| `solve_nivp/moreau_jean_fremond.py` | new | 699 |
| `tests/test_projected_radau_contact.py` | edited | 1 test migrated to `'soc_fb_uniform'` |
| `tests/test_soccp_pgs.py` | new | 174 |
| `tests/test_moreau_jean_fremond.py` | new | 346 |

## What's not done in this session

- **Right-preconditioner on `dΠ` columns** in
  `ProjectedRadauContactModel.step()` — the `1/h` conditioning issue at
  large step sizes flagged earlier in the conversation. The fix is a
  ~5-line column-scaling change in the linear solve inside `step()`;
  not landed yet.
- **Per-stage Jacobian refresh in slip phases** — the rotating-slip-
  direction issue. Requires a one-flag policy change in `step()`.
- **Analytic Clarke selection** for the `pinv(L_s)` boundary case in
  `_soc_fb_phi_and_jac`. Worth doing whenever SOC-FB is used at the
  endpoint on regime-transition problems; not required for the current
  embedded-crack work.
- **Empirical stress test** with `g`/`v_N` decoupling (oscillatory
  normal motion + sustained slip) to validate the velocity-level vs
  position-level Stage 1 trade-off without regime masking.
- **PMMA sliding block reference run** (Acary-Collins-Craft §5.3) and
  **Garagash-style fluid injection on a slip-weakening fault** as
  end-to-end validation cases for the Frémond integrator. Building
  blocks are present; running and committing the notebooks is a
  separate task.

## Recommended next session

1. Apply the `dΠ` right-preconditioner + per-stage Jacobian refresh in
   `projected_radau_contact.py:step()`. Re-run the sliding test and
   check whether the late-time `h ≈ 28.8` failures (obs 5827) are
   resolved.
2. Run the embedded-crack notebook with the new `'soc_fb'` (Breuling
   Stage 1 + De Saxcé endpoint) and confirm the Pollard-Segall
   agreement is preserved.
3. Build the PMMA sliding block validation run for the Frémond
   integrator and commit the result alongside this code.

## Reference

V. Acary, F. Collins-Craft (2025). *A second-order Moreau-Jean scheme
with the Frémond impact law for the Newton-Lagrange formulation of
frictional contact dynamics.* HAL-04230941.

T. Breuling. *Projected Runge-Kutta methods for nonsmooth mechanical
systems.* Dissertation, Stuttgart 2024 (in this repo at
`/home/david/Documents/Friction/Dissertation_Breuling_projected_rk/`).
