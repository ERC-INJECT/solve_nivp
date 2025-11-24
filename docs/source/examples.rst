Examples
========

Sign projection (scalar)
------------------------

.. code-block:: python

    import numpy as np
    from solve_nivp import solve_ivp_ns, SignProjection

    # trivial rhs, demonstrate projection behavior
    rhs = lambda t, y: -y
    t_span = (0.0, 0.2)
    y0 = np.array([0.5, 0.0])  # y, w

    proj = SignProjection(y_indices=0, w_indices=1)

    t, y, h, fk, info = solve_ivp_ns(
        fun=lambda t, s: np.array([-s[0], 0.0]),
        t_span=t_span,
        y0=y0,
        method='backward_euler',
        projection='sign',
        solver='semismooth_newton',
        projection_opts={'y_indices': 0, 'w_indices': 1},
        solver_opts={'globalization': 'linesearch'}
    )

    # y[:,0] is state y, y[:,1] is projected w in sign(y)

Coulomb projection (toy)
------------------------

.. code-block:: python

    import numpy as np
    from solve_nivp import solve_ivp_ns, CoulombProjection

    # toy constraint force: f_conf(y) = K y (diagonal)
    K = np.array([2.0, 3.0, 0.0])
    def con_force(y, t=None, Fk_val=None):
        return K * y

    y0 = np.zeros(6)  # pairs (v_i, z_i)
    t_span = (0.0, 0.1)

    t, y, h, fk, info = solve_ivp_ns(
        fun=lambda t, s: -s,
        t_span=t_span,
        y0=y0,
        method='composite',
        projection='coulomb',
        solver='VI',
        projection_opts={
            'con_force_func': con_force,
            'rhok': np.ones_like(y0),
            'constraint_indices': np.array([0, 2, 4]),
            'conf_jacobian_mode': 'full',
        },
    )

