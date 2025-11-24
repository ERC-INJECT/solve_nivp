---
title: "solve_nivp: A Python toolkit for integrating nonsmooth dynamical systems"
authors:
  - name: David Riley
    affiliation: 1
  - name: Ioannis Stefanou
    affiliation: 1
affiliations:
  - name: IMSIA (UMR 9219), CNRS, EDF, CEA, ENSTA Paris, Institut Polytechnique de Paris, Palaiseau, France
    index: 1
date: "17 October 2025"
bibliography: biblio.bib
---

# Summary

**solve_nivp** is a Python library for time integration of *nonsmooth* ODE/DAE models—systems with abrupt changes such as impacts, switching, or inequality constraints. Such models are widespread: frictional contact in mechanics, piecewise and switching behaviour in circuits, sliding–mode control, and discontinuous rules in finance and energy markets [@brogliato2016nonsmooth; @acary2008numerical; @bernardo2008piecewise; @ShtesselEdwardsFridmanLevant2014; @Tramontana2010; @gabriel2012complementarity]. Classical solvers, which assume smoothness, often require heavy regularisation or very small steps due to the inherent stiffness of these models. **solve_nivp** builds nonsmooth rules directly into the implicit time-stepping scheme, enabling users to encode constraints and advance the state robustly. Documentation, examples, and tests accompany the code for reproducible use.

# Statement of need

Many models in mechanics, circuits, biology, control, and quantitative markets exhibit discontinuities or set-valued relations (ideal diodes, genetic switching, finance applications, stick–slip friction, unilateral contact, switching, saturation). While nonsmooth time-stepping schemes are well established [@StewartTrinkle1996; @acary2008numerical], practitioners working in Python often use interfaces oriented to smooth right-hand sides and event detection. In practice, this pushes users toward ad hoc regularisation or very small time steps to cope with kinks and set-valued relations, thereby increasing computational cost and complicating reproducibility.

Existing libraries such as *Siconos* provide C++ with a Python interface for complementarity-based and rigid-body contact dynamics [@Siconos]. At the other end of the spectrum, general-purpose DAE solvers such as *SUNDIALS/IDA* deliver robust variable-order BDF methods for smooth problems [@hindmarsh2005sundials]. Between these options, there is a gap for a lightweight, Python-native library that lets users encode nonsmooth rules directly and couple them with implicit integrators and globalisation strategies—without adopting a full multibody framework or rewriting models in another language.

**solve_nivp** addresses this need with a minimal projector interface for expressing set-valued relations, together with nonlinear solvers and implicit time-stepping tailored to piecewise-smooth behaviour. The design targets research and prototyping workflows common in computational mechanics and mathematics, robotics, and control: small-to-moderate problem sizes, clear separation of model and solver components, and emphasis on reproducible examples and tests. By bringing projection-based nonsmooth techniques into a familiar scientific Python workflow, the package lowers the barrier to building, comparing, and sharing nonsmooth models alongside existing tools.

# Functionality and design

**solve_nivp** is organised around three interchangeable components. Users encode the nonsmooth rule as a constraint (i.e., a projection onto a convex set), choose a nonlinear solver, and select an implicit integrator. The library wires these components and integrates the ODE/DAE in time while enforcing the constraint at each step, helping keep models readable and making it straightforward to swap algorithms during experimentation.

Projectors are small classes exposing `project()` and an optional generalised derivative via `tangent_cone()`. The distribution includes several built-in projection methods; users can add new ones by implementing the same two methods. For the nonlinear step, the package provides a semismooth Newton method [@qi1993nonsmooth] with Armijo line search [@armijo1966minimization] and a variational-inequality fixed-point iteration [@Uzawa1958; @FortinGlowinski1983], both aimed at nonsmooth problems and offering standard tolerances, safeguards, and iteration diagnostics.

Time integration of the dynamical system is implicit and includes the Backward Euler [@hairer1993solving], Trapezoidal [@hairer1993solving], and $\theta$ methods [@hairer1993solving], as well as a composite scheme based on the Bathe method [@bathe2012insight]. An adaptive controller is available and can exclude algebraic components from the local-error norm, which is useful for mixed ODE/DAE models. An optional reinforcement-learning add-on exposes the time integrator as an environment for learning adaptive step-size policies. In a companion study [@Riley2025RLAdaptive], RL-based policies achieved substantial speedups while maintaining accuracy, demonstrating the approach’s potential for hard, nonsmooth problems. Linear-algebra routines operate on dense or sparse arrays in the SciPy ecosystem [@virtanen2020scipy]. 

The public API exposes both a convenience entry point (`solve_nivp`) and low-level building blocks (projectors, solvers, integrators) to support rapid prototyping and comparative studies. The repository includes examples and notebooks demonstrating contact-rich mechanics and switching problems, together with unit tests and Sphinx documentation, so users can reproduce results with minimal setup.

# Acknowledgements

The authors acknowledge support from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement no. 101087771, INJECT).

# References
