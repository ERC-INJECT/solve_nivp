---
title: "Solve_IVP_NS: Projection-based nonsmooth time integration in Python"
tags:
  - Python
  - nonsmooth dynamics
  - variational inequalities
  - semismooth Newton
  - time integration
authors:
  - given-names: First
    surname: Last
    affiliation: "1"
affiliations:
  - index: 1
    name: Department, Institution, City, Country
date: 12 August 2025
bibliography: paper.bib
---

# Summary

Solve_IVP_NS is a lightweight Python toolkit for integrating nonsmooth ordinary and differential–algebraic equations using projection-based methods. Users express set-valued relations (e.g., sign constraints and Coulomb-like friction) through a simple projector interface—`project()` with an accompanying generalized derivative `tangent_cone()`—and then choose from built-in implicit integrators and nonlinear solvers. The package targets small-to-medium problems where rapid prototyping and domain-specific extensions matter, offering semismooth Newton and variational-inequality iterations with optional line search, adaptive step control, and sparse linear solves. Typical applications include stick–slip friction, sliding-mode control, and rate-independent constitutive updates. Documentation, examples, and unit tests are included to support reproducible use.

# Statement of need

Many dynamical systems feature set-valued relations and kinks (e.g., friction, saturation, elastoplasticity) that challenge traditional smooth integrators. While general-purpose convex solvers can address some time-step subproblems and comprehensive platforms for nonsmooth dynamics exist, there remains a need for a minimal, Python-native workflow that makes it easy to prototype custom nonsmooth laws and integrate them robustly. 

Solve_IVP_NS contributes:

- A projector contract (project, tangent_cone) to encode domain rules succinctly.
- Two complementary nonlinear solvers—variational inequality iterations and semismooth Newton with Armijo line search—suited for piecewise-smooth problems.
- Implicit integrators (Backward Euler, Trapezoidal, Composite, Embedded BETR) with an adaptive controller that can exclude algebraic components from the LTE norm.
- Dense and sparse paths (GMRES/ILU or sparse LU) with low overhead for moderate sizes.

This design complements convex-optimization approaches: users may embed a QP/SOCP inside a projector when a substep is convex, while keeping the time-integration and globalization machinery. Compared to heavier frameworks, the library lowers the barrier to exploring new nonsmooth models and validating them with tests and examples.

# Functionality and design (brief)

The package provides `IdentityProjection`, `SignProjection`, and a Coulomb-like projector with analytical/finite-difference Jacobians; a semismooth Newton solver with assembled-J path and a VI solver; and integrators with adaptive control and diagnostics. The high-level `solve_ivp_ns` function wires projections/solvers/integrators with minimal configuration. Sphinx documentation (quickstart and examples) and continuous integration (tests across Python versions; docs build) are included for reviewer verification.

# Acknowledgements

We thank colleagues and contributors for feedback. This work uses NumPy and SciPy; optional acceleration uses Numba. Any financial support can be acknowledged here.

# References

See the bibliography file for example entries, e.g., [@doe2020] and a software archive entry [@your_software_citation]. Add domain-relevant references here.

---
title: "YourPackage: a short, specific tagline"
tags:
  - Python
  - computational mechanics
  - non-smooth mechananics
  - numerical time integration
authors:
  - name: David M. Riley
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Ioannis Stefanou
    affiliation: "2"
affiliations:
  - name: Your Lab or Department, University, Country
    index: 1
  - name: Collaborating Institution, Country
    index: 2
date: 12 August 2025
bibliography: paper.bib
---

# Summary
Traditional numerical time integration packages rely on the fact that the ordinary differential equation has a continuous time derivate. However, many systems may not exhibit such a feature, e.g., the case of friction, impact or switchs. To address such challenges, this software package is capable of successfully integrating such systems by relying on convex analysis. Therefore, we bypass the need for either regularization or event-based (which captures each independent event).

Furthermore, the software is intended for ease of use for end-based users by mirroring the style of solve_ivp from scipy. The result is a software packacge that handles the implementation of nonsmooth nature allowing end-users to focus solely on 

# Statement of need
Many problems in science, engineering, or even finance exhibit sudden changes in their behavior, making it difficult for traditional methods to predict and simulate their future state accurately. These systems are described by evolution equations that lack regularity and are not analytic, meaning they are inherently non-smooth. The ubiquity of these types of systems is evident as they are observed across various disciplines such as mechanics ~\cite{brogliato1999nonsmooth,acary2008numerical}, electrical circuits~\cite{acary2008numerical,bernardo2008piecewise}, to biology~\cite{casey2006piecewise,bernardo2008piecewise} and control theory (sliding mode controllers/observers)~\cite{b:SMC_Fridman},  as well as in market models for finance~\cite{tramontana2010complicated} or energy~\cite{valencia2020non} to mention a few. 

# Usage
Brief usage example or figure if it helps readers understand scope.

# Acknowledgements
David Riley and Ioannis Stefanou want to acknowledge the European Research Council’s (ERC) support under the European Union’s Horizon 2020 research and innovation programme (Grant agreement no. 101087771 INJECT). 

# References
