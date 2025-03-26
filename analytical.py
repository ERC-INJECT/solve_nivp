import numpy as np

def solve_damped_constant_offset(m, c, k, offset, s0, v0):
    """
    Solve:
        m ddot s + c dot s + k s = offset   (constant offset)
    with initial s(0)=s0, v(0)=v0,
    returning continuous functions s(t), v(t).
    
    offset might be -mu_res or +mu_res etc. 
    """
    w0 = np.sqrt(k/m)
    zeta = c/(2*np.sqrt(m*k))

    # particular solution => s_p = offset / k
    s_p = offset/k

    def underdamped(A, B):
        wd = w0 * np.sqrt(1 - zeta**2)
        def s_fun(t):
            return s_p + np.exp(-zeta*w0*t)*(A*np.cos(wd*t) + B*np.sin(wd*t))
        def v_fun(t):
            # derivative 
            exp_ = np.exp(-zeta*w0*t)
            cos_ = np.cos(wd*t)
            sin_ = np.sin(wd*t)
            part1 = exp_ * ( -A*wd*sin_ + B*wd*cos_ )
            part2 = -zeta*w0 * exp_ * ( A*cos_ + B*sin_ )
            return part1 + part2
        
        def a_fun(t):
            # a(t) = ( offset - c*v(t) - k*s(t) ) / m
            return (offset - c*v_fun(t) - k*s_fun(t))/m
        return s_fun, v_fun, a_fun

    def overdamped(A, B):
        # r1,r2 => solutions
        disc = c**2 - 4*m*k
        r1 = (-c + np.sqrt(disc)) / (2*m)
        r2 = (-c - np.sqrt(disc)) / (2*m)
        def s_fun(t):
            return s_p + A*np.exp(r1*t) + B*np.exp(r2*t)
        def v_fun(t):
            return A*r1*np.exp(r1*t) + B*r2*np.exp(r2*t)
        def a_fun(t):
            return (offset - c*v_fun(t) - k*s_fun(t))/m
        return s_fun, v_fun, a_fun
    
    def critical(A, B):
        # s_h(t)= e^(-w0 t)*(A + B t), with w0= c/(2 m)
        w0_ = c/(2*m)
        def s_fun(t):
            return s_p + np.exp(-w0_*t)*(A + B*t)
        def v_fun(t):
            exp_ = np.exp(-w0_*t)
            return exp_*B - w0_*exp_*(A + B*t)
        def a_fun(t):
            return (offset - c*v_fun(t) - k*s_fun(t))/m
        return s_fun, v_fun, a_fun

    # homogeneous init
    # s(0)= s0 => s_h(0)= s0 - s_p => call that S0
    # v(0)= v0 => v_h(0)= v0
    S0 = s0 - s_p
    z  = zeta
    if abs(z-1.0)<1e-12:
        # critical
        A_ = S0
        w0_ = c/(2*m)
        B_ = v0 + w0_*A_
        return critical(A_, B_)
    elif z<1:
        # underdamped
        wd = w0*np.sqrt(1 - z**2)
        A_ = S0
        B_ = (v0 + z*w0*A_)/wd
        return underdamped(A_, B_)
    else:
        # overdamped
        disc = c**2 - 4*m*k
        r1 = (-c + np.sqrt(disc)) / (2*m)
        r2 = (-c - np.sqrt(disc)) / (2*m)
        # S0= A+B
        # v0= r1 A + r2 B
        # => B= S0 - A => r1 A + r2 (S0 - A)= v0 => A(r1- r2)= v0 - r2*S0 => A= [v0 -r2 S0]/(r1- r2)
        A_ = ( v0 - r2*S0 )/( r1 - r2 )
        B_ = S0 - A_
        return overdamped(A_, B_)


def piecewise_coulomb(m, c, k, mu_res, s0, v0, t_end=6.0, n_steps=300):
    """
    Solve: m ddot s + c dot s + k s = -mu_res sign(v).
    Returns:
        t_array, s_array, v_array, a_array
    for both (m=0) and (m>0) cases.

    M=0 => "massless" special handling (exponential decay).
    M>0 => use solve_damped_constant_offset(...) for each slip phase.
    """
    # ===========================================================
    # 1) SPECIAL CASE: MASSLESS SYSTEM (m=0)
    # ===========================================================
    if np.isclose(m, 0):
        t = np.linspace(0, t_end, n_steps)
        s_eq = mu_res / k  # equilibrium displacement if c·v + k·s = mu_res

        # If initial velocity is nonzero, update s0 from force balance
        if not np.isclose(v0, 0):
            required_sign = -np.sign(v0)  
            # c·v + k·s0 = ± mu_res  => s0 = (±mu_res - c·v)/k
            # but the sign depends on direction of v0
            s0 = (-mu_res*required_sign - c*v0) / k

        # Check if already "stuck"
        # means  |k*s0| <= mu_res => friction can hold it in place
        if abs(k*s0) <= mu_res:
            # stuck for all times => v=0, a=0
            return (t,
                    np.full_like(t, s0),
                    np.zeros_like(t),
                    np.zeros_like(t))

        # Otherwise, it will slip with an exponential decay. 
        # sign depends on whether k*s0 > mu_res or < -mu_res, etc.
        if k*s0 > mu_res:
            # offset is +mu_res => c·v + k·s = mu_res
            # => s(t)= s_eq + (s0 - s_eq)*exp(-k t/c)
            # => v(t)= d/dt => ...
            # => a(t)= d/dt => ...
            s_final = s_eq
            D = (s0 - s_eq)        # the difference
            decay = np.exp(-k*t/c) # common factor

            s = s_eq + D*decay
            v = (-k/c)*D*decay
            a = (k**2 / c**2)*D*decay  # derivative of v(t)= -k/c(D e^{-k t/c}) => 
                                      # a(t)= (k^2/c^2)D e^{-k t/c}

        else:
            # offset is -mu_res => c·v + k·s = -mu_res
            # => s(t)= -s_eq + (s0 + s_eq)*exp(-k t/c)
            s_final = -s_eq
            D = (s0 + s_eq)
            decay = np.exp(-k*t/c)

            s = -s_eq + D*decay
            v = (k/c)*D*decay
            a = -(k**2 / c**2)*D*decay

        # Find first sticking time => check if s(t) is within 1e-6 of s_final
        # If that occurs at index i => s, v, a => constant from that point
        # with velocity=0, acceleration=0
        stick_indices = np.where(np.abs(s - s_final) ==0)[0]
        if stick_indices.size > 0:
            first_stick = stick_indices[0]
            s[first_stick:] = s_final
            v[first_stick:] = 0.0
            a[first_stick:] = 0.0

        return t, s, v, a

    # ===========================================================
    # 2) GENERAL CASE: INERTIAL SYSTEM (m > 0)
    # ===========================================================
    t_array = np.linspace(0, t_end, n_steps)

    # Determine friction sign from initial conditions
    if abs(v0) ==0:
        # velocity near zero => check if friction can hold it
        if abs(k*s0) < mu_res:
            # stuck for entire time => v=0, a=0
            return (t_array,
                    np.full_like(t_array, s0),
                    np.zeros_like(t_array),
                    np.zeros_like(t_array))
        # otherwise friction_sign depends on k*s0
        friction_sign = np.sign(k*s0)
    else:
        friction_sign = np.sign(v0)

    offset = -mu_res * friction_sign

    # (A) FIRST SLIP PHASE:
    # call the 3-function version of solve_damped_constant_offset
    s_fun, v_fun, a_fun = solve_damped_constant_offset(m, c, k, offset, s0, v0)
    s_vals = s_fun(t_array)
    v_vals = v_fun(t_array)
    a_vals = a_fun(t_array)

    # Zero-crossing detection for velocity
    sign_changes = np.where(np.diff(np.sign(v_vals)))[0]
    if len(sign_changes) == 0:
        # no velocity sign change => no slip reversal
        return t_array, s_vals, v_vals, a_vals

    # We do have at least one sign change at index cross_idx
    cross_idx = sign_changes[0]

    # Find the exact crossing time by linear interpolation
    t0, t1 = t_array[cross_idx], t_array[cross_idx+1]
    v0_val, v1_val = v_vals[cross_idx], v_vals[cross_idx+1]
    t_zero = t0 - v0_val*(t1 - t0)/(v1_val - v0_val)
    s_zero = np.interp(t_zero,
                       [t0, t1],
                       [s_vals[cross_idx], s_vals[cross_idx+1]])

    # Check if now friction can hold it => |k*s_zero| <= mu_res => stuck
    if abs(k*s_zero) < mu_res:
        # from t_zero onward => stuck => v=0, a=0, s= constant
        t_rest = t_array[t_array > t_zero]
        # s_final is s_zero
        s_stuck = np.full_like(t_rest, s_zero)
        v_stuck = np.zeros_like(t_rest)
        a_stuck = np.zeros_like(t_rest)

        return (
            # time
            np.concatenate([t_array[:cross_idx+1], [t_zero], t_rest]),
            # displacement
            np.concatenate([
                s_vals[:cross_idx+1],
                [s_zero],
                s_stuck
            ]),
            # velocity
            np.concatenate([
                v_vals[:cross_idx+1],
                [0.0],
                v_stuck
            ]),
            # acceleration
            np.concatenate([
                a_vals[:cross_idx+1],
                [0.0],
                a_stuck
            ])
        )
    else:
        # Slip reversal => sign flips from friction_sign to -friction_sign
        new_sign = -friction_sign
        offset2 = -mu_res * new_sign
        # define sub-time array after t_zero
        t_sub = np.linspace(t_zero, t_end, n_steps - cross_idx)

        # (B) SECOND SLIP PHASE:  s2_fun, v2_fun, a2_fun from new offset
        s2_fun, v2_fun, a2_fun = solve_damped_constant_offset(m, c, k, offset2, s_zero, 0.0)
        s2_vals = s2_fun(t_sub[1:] - t_zero)  # skip the 0 index to avoid duplication
        v2_vals = v2_fun(t_sub[1:] - t_zero)
        a2_vals = a2_fun(t_sub[1:] - t_zero)

        # Combine
        return (
            np.concatenate([
                t_array[:cross_idx+1],
                t_sub[1:]
            ]),
            np.concatenate([
                s_vals[:cross_idx+1],
                s2_vals
            ]),
            np.concatenate([
                v_vals[:cross_idx+1],
                v2_vals
            ]),
            np.concatenate([
                a_vals[:cross_idx+1],
                a2_vals
            ])
        )


# Example usage for both cases
if __name__ == "__main__":
    # Case 1: Massless system (m=0)
    t1, s1, v1, a1 = piecewise_coulomb(m=0, c=0.2, k=1.0, mu_res=0.2, s0=0.5, v0=0.0)
    
    # Case 2: Inertial system (m>0)
    t2, s2, v2, = piecewise_coulomb(m=1.0, c=0.1, k=2.0, mu_res=0.0, s0=0.0, v0=0.1)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Massless case
    plt.subplot(221)
    plt.plot(t1, s1, 'b-', label='Displacement (m=0)')
    plt.ylabel('s(t)')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(t1, v1, 'r-', label='Velocity (m=0)')
    plt.ylabel('$\dot{s}(t)$')
    plt.legend()
    
    # Inertial case
    plt.subplot(223)
    plt.plot(t2, s2, 'g-', label='Displacement (m>0)')
    plt.xlabel('Time')
    plt.ylabel('s(t)')
    plt.legend()
    
    plt.subplot(224)
    plt.plot(t2, v2, 'm-', label='Velocity (m>0)')
    plt.xlabel('Time')
    plt.ylabel('$\dot{s}(t)$')
    plt.legend()
    
    plt.tight_layout()
    plt.show()