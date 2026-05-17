# SBM Contact Trace Debug Notes

Date: 2026-05-09

## Context

We were discussing the shifted-boundary / Taylor trace construction for the
embedded crack Mohr-Coulomb convergence test. The observed symptom is a
convergence plateau even though the contact algebra appears clean.

The current shifted nodal trace idea is:

```text
surrogate node x_tilde
project to true crack x_hat
evaluate trace by Taylor extension:

u_hat ~= u(x_tilde) + Delta . grad(u)

where Delta = x_hat - x_tilde
```

The proposed next diagnostic was:

```text
add a permanent geometry-admissibility audit to the convergence driver
and test a trace construction that integrates on the true crack through
cut elements, rather than using nodal Taylor extrapolation from the
surrogate path.
```

## Distance vs Element Containment

`|Delta| = 0.5h` does not by itself mean the projected point is inside the
same incident element. `0.5h` is only a distance measure. Element containment
is a side-of-edges / barycentric-coordinate test.

For a triangle whose crack-side facet is edge `AB`, a point can move a very
small distance across edge `AB` and immediately leave that triangle. In
barycentric coordinates, this shows up as one negative barycentric component.

However, we clarified that this may not be the main issue here: even if
`x_hat` leaves the selected incident element, it can still lie inside another
neighboring element. So "outside one incident element" is not automatically
the reason for the convergence plateau.

## Issue 3: Element Choice for the Taylor Gradient

The more subtle issue is the element used to evaluate `grad(u)` in:

```text
Delta . grad(u)
```

At a mesh node, `grad(u)` is not unique. The node belongs to several triangles,
and for a finite element field each incident triangle has its own local
polynomial gradient.

The current nodal Taylor implementation builds a map like:

```text
scalar_dof -> first element containing that DOF
```

So at a crack node the Taylor correction can use the first incident element
found by DOF ordering, not necessarily:

```text
the plus/minus crack-facet parent element
```

and not necessarily:

```text
the element containing the projected point x_hat
```

This creates a possible consistency mismatch:

```text
Delta:        from surrogate node to true crack
normal:       from true crack / crack-facet pairing
weight:       from surrogate crack facet
u(x_tilde):   from plus/minus duplicated crack DOF
grad(u):      from first incident element chosen by DOF ordering
```

That last line is issue 3.

This does not show up in a global linear patch test, because every element
reproduces the same global linear gradient. But the actual slip/contact field
is not globally linear, especially near tips, localized slip gradients, and
active-set transitions. There, two neighboring incident elements can produce
different `grad(u)`, and the Taylor term can become sensitive to the arbitrary
element choice.

## Important Clarification

Issue 3 does not prove non-convergence by itself.

If `x_hat` is inside another valid same-side element, then a clean SBM method
could still converge, perhaps with reduced order. So the issue is not simply:

```text
x_hat is outside the mesh
```

or:

```text
x_hat is outside every valid element
```

The concern is instead:

```text
the code may be using the wrong local polynomial branch for grad(u)
in the Taylor trace.
```

Earlier testing reportedly forced the crack-facet parent element. That
improved `N=40` but did not remove the plateau. That suggests issue 3 may be
part of the fragility, but probably is not the whole explanation.

## Deeper Suspect

The deeper possible problem is the hybrid formulation:

```text
surrogate-crack nodal contact
+ Taylor extrapolated true-crack trace
+ nonsmooth Coulomb projection
```

The contact law sees nodal shifted traces, not a weak/mortar/cut-interface
trace integrated directly on the true crack. This can be fragile for nonsmooth
contact, especially when `|Delta|/h` is an O(1) fraction of the cell size.

The cosine/orientation correction fixes measure/orientation consistency in
the interface integral. It does not fix trace extrapolation.

## Diagnostic Tests To Run

Compare the following trace variants:

1. Current nodal Taylor method using the first incident element.
2. Nodal Taylor method forcing the crack-facet parent element.
3. Nodal Taylor method using the side-consistent element that contains
   `x_hat`, when available.
4. `l2_project` Taylor method, which replaces the single-element nodal
   gradient with an L2-projected weak gradient.
5. A diagnostic true-crack quadrature/cut-trace method, where contact is
   assembled at quadrature points on the actual crack rather than at
   surrogate nodes shifted by Taylor extrapolation.

If variants 1-4 all plateau, then the failure is probably not only element
choice. The suspect shifts to the nodal shifted-boundary contact formulation
itself.

## Permanent Audit To Add

For each contact node, record:

```text
node index
true-crack coordinate x_hat
surrogate coordinate x_tilde
Delta
|Delta| / h
Delta normal/tangential components
selected Taylor element
crack-facet parent element
element containing x_hat, if any
barycentric coordinates of x_hat in candidate elements
whether any incident side element contains x_hat
local slip error
contact class / active set
```

Then read convergence results together with geometry admissibility. The key
question is whether the Taylor element choice or shifted nodal trace produces
a non-decaying perturbation in local gap/slip.

