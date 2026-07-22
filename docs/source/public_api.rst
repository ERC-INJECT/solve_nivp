Public API Policy
=================

This page defines the supported surface of ``solve_nivp``.  It is intended to
keep the package usable as a general library while still allowing active
research backends to evolve.

Stable API
----------

The stable API is the recommended import surface for users and downstream
projects.  These names should be documented, covered by tests, and changed only
with a compatibility plan.

Top-level convenience API:

* ``solve_nivp.solve_nivp``
* ``solve_nivp.__version__``

System and driver classes:

* ``solve_nivp.ODESystem``
* ``solve_nivp.ODESolver``

Nonlinear solver:

* ``solve_nivp.ImplicitEquationSolver``

Integrator classes:

* ``solve_nivp.BackwardEuler``
* ``solve_nivp.Trapezoidal``
* ``solve_nivp.ThetaMethod``
* ``solve_nivp.CompositeMethod``
* ``solve_nivp.EmbeddedBETR``
* ``solve_nivp.SDIRK2``
* ``solve_nivp.RadauIIA``

Projection classes:

* ``solve_nivp.Projection``
* ``solve_nivp.IdentityProjection``
* ``solve_nivp.SignProjection``
* ``solve_nivp.CoulombProjection``
* ``solve_nivp.GeneralMoreauVIProjection``
* ``solve_nivp.MuScaledSOCProjection``
* ``solve_nivp.MoreauSOCProjection``
* ``solve_nivp.AnisotropicSOCProjection``
* ``solve_nivp.AlgebraicConstraintProjection``
* ``solve_nivp.CompositeContactProjection``

Experimental API
----------------

The experimental API is useful and importable, but it is still allowed to
change while the package design is being refined.  Where possible, changes
should still be documented in release notes and tested against the examples
that use them.

Current experimental areas:

* Contact-system builders and backend-specific helpers.
* Schur-complement contact solvers.
* RATTLE contact-system helpers.
* PETSc-specific large-scale solver paths.
* ``solve_nivp.rl`` reinforcement-learning wrappers.

Private Internals
-----------------

Names beginning with ``_`` are private unless explicitly documented here or in
the API reference.  Internal caches, factorization reuse details, helper
dispatch functions, notebook utilities, and generated example artifacts are not
part of the compatibility contract.

Research And Workspace Material
-------------------------------

The repository currently contains research notebooks, benchmark outputs,
diagnostic scripts, and generated result files.  Those are valuable for
reproducibility, but they are not the core package API.  They are kept outside
the installable library path and should be treated as examples, benchmarks,
external reproductions, or archived artifacts.

Compatibility Rule
------------------

Stable API changes should prefer one of these routes:

* preserve the existing import and behavior;
* add a new explicit option while keeping the old default;
* deprecate first, remove later;
* document unavoidable breaking changes with migration notes.
