# Optional acceleration imports: try to import JAX.
try:
    import jax
    import jax.numpy as jnp
    from jax.config import config
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
    print('JAX_AVAILABLE', JAX_AVAILABLE)
except ImportError:
    JAX_AVAILABLE = False

# Try to import autograd for automatic differentiation.
try:
    from autograd import jacobian as autograd_jacobian
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False

import numpy as np
from scipy.optimize import root
import warnings

class ImplicitEquationSolver:
    """
    Generic solver for the implicit equation F(y)=0.
    
    This solver returns a tuple:
         (y_new, Fk, error, success, iterations)
    
    Supported methods:
      - 'root'
      - 'newton_raphson'
      - 'PEG'
      - 'VI'
      - 'semismooth_newton'
      
    For the semismooth newton method, a projection operator (proj) and functions
    g_func and J_g_func must be provided.
    
    Parameters:
      method : str, default 'root'
          The solver method.
      jacobian : callable, optional
          A user-supplied function to compute the Jacobian.
      tol : float, default 1e-10
          Convergence tolerance.
      max_iter : int, default 100
          Maximum number of iterations.
      proj : object, optional
          Projection operator (must implement .project and .tangent_cone).
      rho0 : float, default 0.9
          Parameter for VI/PEG methods.
      delta : float, default 0.7
          Parameter for VI/PEG methods.
      component_slices : list, optional
          List of slices for partitioning the state (required for some methods).
      use_autodiff : bool, default False
          Whether to use automatic differentiation for Jacobian computation.
      autodiff_mode : str, default 'autograd'
          Which autodiff backend to use: 'jax' (if available) or 'autograd'.
      L : float, default 0.9
          Parameter used in VI line-search.
      Lmin : float, default 0.3
          Lower bound parameter in VI line-search.
      nu : float, default 0.66
          Factor used in VI line-search.
      lam : float, default 1.0
          Parameter used in the semismooth function.
      g_func : callable, optional
          Function used in semismooth newton (typically the integrator residual).
      J_g_func : callable, optional
          User-provided Jacobian for g_func (if autodiff is not used).
    """
    def __init__(self, method='root', jacobian=None, tol=1e-10, max_iter=100,
                 proj=None, rho0=0.9, delta=0.7, component_slices=None,
                 use_autodiff=False, autodiff_mode='autograd', L=0.9, Lmin=0.3,
                 nu=0.66, lam=1.0, g_func=None, J_g_func=None):
        self.method = method
        self.jacobian = jacobian
        self.tol = tol
        self.max_iter = max_iter
        self.proj = proj
        self.rho0 = rho0
        self.delta = delta
        self.component_slices = component_slices
        self.use_autodiff = use_autodiff
        self.autodiff_mode = autodiff_mode.lower()
        self.L = L
        self.Lmin = Lmin
        self.nu = nu
        self.lam = lam  # Parameter for semismooth function

        # Require projection and component slices for PEG, VI, and semismooth_newton.
        if self.method in ['PEG', 'VI'] and self.proj is None:
            raise ValueError(f"Projection operator 'proj' must be provided for method '{self.method}'.")
        if self.method in ['PEG', 'VI'] and self.component_slices is None:
            raise ValueError(f"component_slices must be provided for method '{self.method}'.")
        if self.method == 'semismooth_newton':
            if self.proj is None:
                raise ValueError("Projection operator 'proj' must be provided for semismooth newton method.")
            # If g_func or J_g_func are provided, you may store them.
            if g_func is not None:
                self.g_func = g_func
            if J_g_func is not None:
                self.J_g_func = J_g_func

        # Set up automatic jacobian if requested.
        if self.use_autodiff and self.jacobian is None:
            print(f'JAX_AVAILABLE {JAX_AVAILABLE}')
            if self.autodiff_mode == 'jax' and JAX_AVAILABLE:
                self.jacobian = jax.jit(jax.jacfwd(self._func_wrapper))
            elif self.autodiff_mode == 'autograd' and AUTOGRAD_AVAILABLE:
                self.jacobian = autograd_jacobian(self._func_wrapper)
            else:
                warnings.warn("Requested autodiff backend is not available; falling back to numerical Jacobian.")
                self.jacobian = None  # Will use numerical approximation later


    def _func_wrapper(self, y):
        """
        Wrapper to call the user-supplied function.
        This is used to build the autodiff Jacobian.
        """
        return self.func(y)

    def set_func(self, func):
        """
        Set the function F(y) that is to be solved.
        Attempt to set up autodifferentiation for the Jacobian.
        If an error occurs (e.g. due to JAX tracer issues), fall back to numerical differentiation.
        """
        self.func = func
        if self.use_autodiff and self.jacobian is None:
            try:
                if self.autodiff_mode == 'jax' and JAX_AVAILABLE:
                    # Try to set up JAX autodiff.
                    self.jacobian = jax.jit(jax.jacfwd(self._func_wrapper))
                elif self.autodiff_mode == 'autograd' and AUTOGRAD_AVAILABLE:
                    self.jacobian = autograd_jacobian(self._func_wrapper)
            except Exception as e:
                warnings.warn(f"Autodiff backend {self.autodiff_mode} failed with error: {e}. "
                              "Falling back to numerical Jacobian.")
                self.jacobian = None

    def solve(self, func, y0):
        """
        Solve F(y)=0 for y, starting from initial guess y0.
        
        Returns:
          (y_new, Fk, error, success, iterations)
        """
        self.set_func(func)
        if self.method == 'root':
            return self._solve_with_root(func, y0)
        elif self.method == 'newton_raphson':
            return self._solve_with_newton_raphson(func, y0)
        elif self.method == 'PEG':
            return self._solve_with_PEG(func, y0)
        elif self.method == 'VI':
            return self._solve_with_VI(func, y0)
        elif self.method == 'semismooth_newton':
            return self._solve_with_semismooth_newton(func, y0)
        elif self.method == 'IPRG':
            return self._solve_with_IPRG(func, y0)
        elif self.method == 'EGA':
            return self._solve_with_EGA(func, y0)            
        else:
            raise ValueError(f"Unknown solver method: {self.method}")

    def _solve_with_root(self, func, y0):
        sol = root(func, y0, method='hybr')
        Fk = sol.fun
        success = sol.success
        err = np.linalg.norm(Fk)
        iters = getattr(sol, 'nfev', 1)
        return (sol.x, Fk, err, success, iters)

    def _solve_with_newton_raphson(self, func, y0):
        y = y0.copy()
        for iteration in range(self.max_iter):
            F = func(y)
            errF = np.linalg.norm(F, ord=2)
            if errF < self.tol:
                return (y, F, errF, True, iteration+1)
            # Compute Jacobian either via autodiff or numerical finite-differences.
            if self.jacobian is not None:
                J = self.jacobian(y)
            else:
                J = self._numerical_jacobian(func, y)
            try:
                delta_val = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return (y, F, errF, False, iteration+1)
            y_new = y + delta_val
            if np.linalg.norm(delta_val, ord=2) < self.tol:
                F_new = func(y_new)
                err_new = np.linalg.norm(F_new)
                return (y_new, F_new, err_new, True, iteration+1)
            y = y_new
        F_final = func(y)
        return (y, F_final, np.linalg.norm(F_final), False, self.max_iter)

    def _solve_with_semismooth_newton(self, func, y0):
        """
        Modified semismooth Newton method that:
          1) Evaluates the residual F_in = func(y)
          2) Forms candidate = y - lam * F_in
          3) Projects the candidate using proj.project
          4) Defines the semismooth function F = y - proj(candidate)
          5) Builds a semismooth Jacobian: J = I - Dproj*(I - lam*J_in)
          6) Solves J * delta = -F via linear solve.
        
        Returns:
          (projected_solution, F_in, error, success, iterations)
        """
        y = y0.copy()
        lam = self.lam
        n = len(y)
        I = np.eye(n)
        # Track initial residual for relative tolerance
        # initial_res = np.linalg.norm(func(y))
        
        for iteration in range(self.max_iter):
            F_in = func(y)  # Evaluate residual
            candidate = y - lam * F_in  # Full Newton candidate
            proj_z = self.proj.project(y, candidate, rhok=None)  # Apply projection
            F = y - proj_z  # Semismooth function
            errF = np.linalg.norm(F)
            if errF < self.tol:
                return (proj_z, F_in, errF, True, iteration+1)
            # Compute derivative of the residual (J_in) via autodiff or finite differences.
            if self.jacobian is not None:
                J_in = self.jacobian(y)
            else:
                J_in = self._numerical_jacobian(func, y)
            # Get derivative of the projection: tangent cone matrix.
            Dproj = self.proj.tangent_cone(candidate, proj_z)
            J = I - Dproj @ (I - lam * J_in)
            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return (y, F, errF, False, iteration+1)
            y_new = y + delta
            # if np.linalg.norm(delta) < self.tol:
            #     F_new = func(y_new)
            #     candidate_new = y_new - lam * F_new
            #     proj_new = self.proj.project(y_new, candidate_new, rhok=None)
            #     F_smooth = y_new - proj_new
            #     err_new = np.linalg.norm(F_smooth)
            #     return (proj_new, F_new, err_new, True, iteration+1)
            y = y_new
        return (proj_z, F_in, errF, False, self.max_iter)

    # def _solve_with_PEG(self, func, y0):
    #     x = y0.copy()
    #     x_ = y0.copy()
    #     Fx = func(x)
    #     k = 1
    #     success = False
    #     con = self.delta + self.delta**2
    #     num_comps = len(self.component_slices)
    #     rhok = np.full(num_comps, self.rho0)
    #     thetak_minus1 = np.ones(num_comps)
    #     y_new = x.copy()
    #     while k <= self.max_iter:
    #         adjustment = np.zeros_like(x)
    #         for i, sl in enumerate(self.component_slices):
    #             adjustment[sl] = rhok[i] * Fx[sl]
    #         y_new = self.proj.project(x, x_ - adjustment, rhok)
    #         Fy = func(y_new)
            
    #         for i, sl in enumerate(self.component_slices):
    #             diff_y = y_new[sl] - x[sl]
    #             diff_f = Fy[sl] - Fx[sl]
    #             ndy2 = np.dot(diff_y, diff_y)
    #             ndf2 = np.dot(diff_f, diff_f)
    #             if ndf2 >=  1e-12 and ndy2 >=1e-12:
    #                 temp = thetak_minus1[i] / (4 * rhok[i] * self.delta)
    #                 lambda_candidate = temp * (ndy2 / ndf2)
    #                 rho_k_new = min(con * rhok[i], lambda_candidate)
    #             else:
    #                 rho_k_new = rhok[i]


    #             denom_val =  rhok[i] * self.delta
    #             thetak = rho_k_new / (rhok[i] * self.delta)
    #             thetak_minus1[i] = thetak

    #             if np.abs(denom_val) < 1e-12 or np.isnan(denom_val) or np.isnan(thetak_minus1[i]):
    #                 print(f"At iteration {k}, index {i}: thetak_minus1 = {thetak_minus1[i]}, denominator = {denom_val}, ndy2: {ndy2}, rho value: {rhok[i]}")
    #             rhok[i] = max(rho_k_new,1e-12)
    #         for i, sl in enumerate(self.component_slices):
    #             x_[sl] = (1 - self.delta) * y_new[sl] + self.delta * x_[sl]
    #         err_local = 0.0
    #         for sl in self.component_slices:
    #             denom = np.linalg.norm(y_new[sl]) + 1e-10
    #             err_local = max(err_local, np.linalg.norm(y_new[sl] - x[sl]) / denom)
    #         if err_local < self.tol:
    #             success = True
    #             break
    #         # if success == False:
    #         #     return  (y0, func(y0), err_local, success, k)
    #         x = y_new.copy()
    #         Fx = Fy.copy()
    #         k += 1
    #     return (y_new, Fy, err_local, success, k)


    def _solve_with_PEG(self, func, y0):
        """
        PEG method with a single (global) rhok for all slices, 
        and an error measure scaled by sqrt(N).

        Returns:
            (y_new, Fy, err_local, success, iterations)
        """
        # Initialization
        x = y0.copy()
        x_ = y0.copy()
        Fx = func(x)                      # initial residual
        k = 1
        success = False

        # For the step-size adaptation
        con = self.delta + self.delta**2  # e.g. delta=0.7 => con=1.19
        rhok = self.rho0                  # single scalar
        thetak_minus1 = 1.0

        y_new = x.copy()
        
        # Precompute normalizing factor for the error measure
        # or compute each iteration if problem size might change
        Nsqrt = np.sqrt(x.size)

        while k <= self.max_iter:
            # 1) Weighted step for the entire vector
            adjustment = rhok * Fx
            
            # If self.proj.project expects an array for rhok, make one of length #slices
            # (each slice sees the same rhok).
            # If your 'projection' is fine with a scalar, just pass rhok directly.
            big_rhok = np.full(len(self.component_slices), rhok)

            # 2) Projection
            y_new = self.proj.project(x, x_ - adjustment, big_rhok)
            Fy = func(y_new)

            # 3) Single scalar step-size update logic
            diff_y = y_new - x
            diff_f = Fy - Fx
            ndy2 = np.dot(diff_y, diff_y)
            ndf2 = np.dot(diff_f, diff_f)

            if (ndf2 >= 1e-12):
                temp = thetak_minus1 / (4.0 * rhok * self.delta)
                lambda_candidate = temp * (ndy2 / ndf2)
                rhok_new = min(con * rhok, lambda_candidate)
            else:
                rhok_new = rhok

            thetak = rhok_new / (rhok * self.delta)
            thetak_minus1 = thetak
            
            # safeguard
            rhok = max(rhok_new, 1e-12)

            # 4) Momentum update for x_
            x_ = (1.0 - self.delta)*y_new + self.delta*x_

            # 5) Convergence check
            # If you want the same style as your other methods 
            # but scale by sqrt(N) so that errors don't scale with dimension:
            err_local = np.linalg.norm(y_new - x) / Nsqrt

            if err_local < self.tol:
                success = True
                break

            # prepare for next iteration
            x = y_new.copy()
            Fx = Fy.copy()
            k += 1

        return (y_new, Fy, err_local, success, k)


    def _solve_with_EGA(self, func, y0):
        """
        EG–Anderson(1) method, implementing precisely the steps from the
        snippet:

        Algorithm 1: EG–Anderson(1):
        1) Initialization: 
            - choose x0 in Omega
            - omega >= 0, gamma > 0, tau > 1/2, rho, mu in (0,1)
            - sigma0 = 1, M >= 1
            - a sufficiently large M>0
            For k=0,1,2... do:

        Step 1: compute F_y(xk) = P( xk, xk - gamma * H(xk), gamma ) - xk
                if F_y(xk)=0 => stop
                otherwise set tk = gamma, go to step2
        Step 2: yk+0.5 = P( xk, xk - tk * H(xk), tk )
                yk+1   = P( xk, xk - tk * H( yk+0.5 ), tk )
                if    tk * <H(yk+0.5)-H(xk),  yk+0.5 - yk+1>
                        <= mu/2 ( ||xk-yk+0.5||^2 + ||yk+0.5-yk+1||^2 )
                        => go step3
                else  tk = rho * tk, repeat step2
        Step 3: F_tk(xk)     = yk+0.5 - xk
                Ftilde_tk(xk)= yk+1 - xk
                if ||Ftilde_tk|| < min( ||F_tk||, omega*sigma_k^(-tau) ) 
                    => alpha_k=  <Ftilde, Ftilde-F_tk>/||Ftilde-F_tk||^2
                else alpha_k= M+1
        Step 4: if |alpha_k|<=M => x_{k+1}= alpha_k * xk + (1-alpha_k)*yk+1
                                        sigma_{k+1}= sigma_k+1
                else => x_{k+1}= yk+1
                        sigma_{k+1}= sigma_k
        (end for)

        We treat "H(x)" as your 'func(x)'. The projection is done via self.proj.project(...).

        Returns
        -------
        (x_final, F_res, err, success, iterations)
        where F_res = P(...) - x_final, the final residual
        """
        import numpy as np

        max_iter = self.max_iter
        tol      = self.tol

        # -------------- Retrieve or default the needed parameters --------------
        # The snippet uses parameters:
        #   gamma  > 0   (the initial step-size, or line-search base)
        #   rho    in (0,1) (backtracking scale)
        #   mu     in (0,1) (the line-search acceptance parameter)
        #   tau    > 1/2
        #   omega  >= 0
        #   sigma0 = 1
        #   M >= 1
        # You can store them in the solver or pass them via solver_opts.
        gamma  = getattr(self, "gamma", 0.1)  # or from self.solver_opts
        rho    = getattr(self, "rho",   0.8)
        mu     = getattr(self, "mu",    0.5)
        tau    = getattr(self, "tau",   0.6)
        omega  = getattr(self, "omega", 30.0)
        M      = getattr(self, "M",     5000)

        # Initialize
        xk     = y0.copy()
        sigma_k= 1.0  # per snippet: sigma0=1
        iteration=0
        success = False

        # -------------- Step 1: compute F_y(xk) = P(...) - xk --------------
        def F_y(x):
            # projection = POmega(xk - gamma * H(xk))
            proj_x = self.proj.project(x, x - gamma*func(x), gamma)
            return proj_x - x

        Fy_k = F_y(xk)
        err  = np.linalg.norm(Fy_k)
        if err < tol:
            # done immediately
            return (xk, func(xk), err, True, iteration)

        # main loop
        while iteration < max_iter:
            iteration += 1

            # # Step1 re-check: if Fy_k=0 => done
            # if err < tol:
            #     success = True
            #     break

            # Otherwise set tk= gamma, go to Step2
            tk = gamma

            # Step2: do a small backtracking loop
            #   yk+0.5= P( xk, xk - tk*H(xk), tk )
            #   yk+1  = P( xk, xk - tk*H(yk+0.5), tk )
            #   check condition (3.1):
            #    tk * < H(yk+0.5)-H(xk),  yk+0.5-yk+1 >
            #    <= mu/2 ( ||xk-yk+0.5||^2 + ||yk+0.5-yk+1||^2 )
            backtrack_ok = False
            max_backtrack=20000

            for _ in range(max_backtrack):
                # yk+0.5
                y_half = self.proj.project(xk, xk - tk*func(xk), tk)
                # yk+1
                y_next = self.proj.project(xk, xk - tk*func(y_half), tk)

                # condition (3.1)
                lhs_vec = func(y_half) - func(xk)
                lhs     = tk * np.dot(lhs_vec, (y_half - y_next))
                # the right side
                rside= 0.5*mu*( np.linalg.norm(xk - y_half)**2
                                + np.linalg.norm(y_half - y_next)**2 )
                if lhs <= rside:
                    backtrack_ok= True
                    break
                else:
                    tk = rho*tk

            # if we never satisfied => proceed with minimal tk
            if not backtrack_ok:
                y_half = self.proj.project(xk, xk - tk*func(xk), tk)
                y_next = self.proj.project(xk, xk - tk*func(y_half), tk)

            # Step3: define F_tk(xk), Ftilde_tk(xk)
            F_tk      = y_half - xk     # (yk+0.5 - xk)
            Ftilde_tk = y_next - xk     # (yk+1 - xk)

            normF    = np.linalg.norm(F_tk)
            normFtld = np.linalg.norm(Ftilde_tk)

            # condition => if ||Ftilde_tk|| < min( ||F_tk||,  omega*sigma_k^-tau )
            #   alpha_k= <Ftilde_tk, Ftilde_tk-F_tk> / || Ftilde_tk-F_tk||^2
            # else alpha_k= M+1
            alpha_k = M+1  # default
            # check if normFtld < min(normF,  omega*(sigma_k^(-tau)) )
            if normFtld < min( normF, omega*(sigma_k**(-tau)) ):
                # alpha_k= <Ftilde_tk, (Ftilde_tk - F_tk)> / ||Ftilde_tk - F_tk||^2
                diff = Ftilde_tk - F_tk
                denom= np.dot(diff, diff)
                if denom > 1e-15:  # avoid zero division
                    alpha_k= np.dot(Ftilde_tk, diff)/ denom

            # Step4:
            # if |alpha_k| <= M => x_{k+1}= alpha_k xk + (1-alpha_k) yk+1, sigma_{k+1}= sigma_k+1
            # else => x_{k+1}= yk+1, sigma_{k+1}= sigma_k
            if abs(alpha_k) <= M:
                xkp1 = alpha_k*xk + (1.0-alpha_k)*y_next
                sigma_new= sigma_k+1
            else:
                xkp1= y_next
                sigma_new= sigma_k

            # update xk => xkp1
            xk = xkp1
            sigma_k= sigma_new

            # compute final residual for next iteration
            # Fy_k= self.proj.project(xk, xk - gamma*func(xk), gamma) - xk
            temp = self.proj.project(xk, xk - func(xk), gamma/gamma)
            err = np.linalg.norm(xk -temp )
            if err < tol:
                xk = temp
                success= True
                break

        # End while
        return (xk, func(xk), err, success, iteration)


    def _solve_with_IPRG(self, func, y0):
        """
        Inertial Projected Reflected Gradient (IPRG) method.

        Implements the steps:

          Given x_{-1}, x0, lam0 > 0, theta in [0, 1/7], and mu in (0, mu-bar):

            1) w_n = x_n + theta * (x_n - x_{n-1})
               y_n = 2*x_n - x_{n-1}
               x_{n+1} = P_C( w_n - lam_n * A(y_n) )

               if w_n == y_n == x_{n+1}, stop

            2) dot_n = < A(y_n) - A(y_{n-1}), y_n - x_{n+1} >
               if dot_n <= 0:
                   lam_{n+1} = lam_n
               else:
                   rho_n = sqrt( 2||y_{n-1} - x_n||^2
                                 + (2+sqrt{2})||x_n - y_n||^2
                                 + 2||x_{n+1} - y_n||^2 )
                   lam_{n+1} = min( mu_iprg * rho_n / dot_n, lam_n )

            3) n <- n+1, go to 1)

        We interpret 'func' as the operator A(\cdot).
        We use 'self.proj.project(...)' for the projection P_C.

        Returns
        -------
        (x_final, Fk, err, success, iterations)

        where:
          - x_final = x_{n+1} upon termination
          - Fk = func(x_final)
          - err = final || x_{n+1} - x_n ||
          - success = bool
          - iterations = number of iterations performed
        """

        # We need an initial x_{-1} and x_0.
        # If the user only gives one guess y0, we will just set x_{-1} = x_0 = y0
        x_m1 = y0.copy()  # x_{-1}
        x_n  = y0.copy()  # x_0

        # lam_n  = self.lam0     # initial lambda_0
        # theta  = self.theta    # inertial param
        # mu_val = self.mu_iprg  # "mu" in the paper
        # Initialize parameters (θ, μ, λ₀)
        theta = (0.99 / 7) #self.theta if self.theta is not None else (0.99 / 7)
        lam_n = 1.0 #self.lambda0  # λ₀ = 1.0
        mu_bar = (1 - 7 * theta) / (4 + 2 * np.sqrt(2))  # μ̄ formula
        mu_val = 0.99 * mu_bar #self.mu if self.mu is not None else 0.99 * mu_bar

        # We'll store y_{n-1} = 2*x_{n-1} - x_{n-2}, but for the first iteration
        # we do not have x_{n-2}. We'll just skip the dot-product step the very first time.
        # We keep track of y_old = y_{n-1}, A(y_old) from the previous iteration.
        y_old = None
        Ay_old = None

        success = False
        for iteration in range(1, self.max_iter+1):
            # Step 1: form w_n, y_n
            w_n = x_n + theta*(x_n - x_m1)
            y_n = 2.0*x_n - x_m1

            # compute the candidate
            A_y_n = func(y_n)
            x_np1 = self.proj.project(w_n, w_n - lam_n*A_y_n, lam_n)

            # Check the "stop if w_n == y_n == x_{n+1}" in a numerical sense:
            if (np.allclose(w_n, y_n, rtol=1e-12, atol=1e-15)
                and np.allclose(y_n, x_np1, rtol=1e-12, atol=1e-15)):
                success = True
                break

            # Step 2: update lambda_{n+1}, but only if we have y_{n-1}
            # i.e. only if iteration >= 2
            if y_old is not None:
                # dot_n = < A(y_n) - A(y_{n-1}), y_n - x_{n+1} >
                diff_Ay = A_y_n - Ay_old
                diff_yx = y_n - x_np1
                dot_n = np.dot(diff_Ay, diff_yx)

                if dot_n <= 0.0:
                    lam_next = lam_n
                else:
                    # Compute rho_n
                    #   rho_n = sqrt( 2||y_{n-1}-x_n||^2 + (2+sqrt{2})||x_n-y_n||^2
                    #                 + 2||x_{n+1}-y_n||^2 )
                    term1 = 2.0*np.linalg.norm(y_old - x_n)**2
                    term2 = (2.0+np.sqrt(2.0))*np.linalg.norm(x_n - y_n)**2
                    term3 = 2.0*np.linalg.norm(x_np1 - y_n)**2
                    rho_n =term1 + term2 + term3
                    lam_next = min(mu_val * rho_n / dot_n, lam_n)
            else:
                # No update if it's the very first iteration
                lam_next = lam_n

            # Step 3: shift indices:
            #   x_{-1} <- x_n
            #   x_n    <- x_{n+1}
            #   y_{n-1} <- y_n, etc.
            x_m1 = x_n
            x_n  = x_np1
            y_old   = y_n
            Ay_old  = A_y_n
            lam_n   = lam_next

            # Check for convergence in the usual sense:
            err = np.linalg.norm(x_n - x_m1)
            if err < self.tol:
                success = True
                break

        # End of loop
        Fk = func(x_n)
        # final error measure
        err_final = np.linalg.norm(x_n - x_m1)

        return (x_n, Fk, err_final, success, iteration)

    def _solve_with_VI(self, func, y0):
        k = 0
        yk = y0.copy()
        rho = self.rho0
        Fk_val = func(yk)
        y_proj = self.proj.project(yk, yk - rho * Fk_val, rho)
        err = np.linalg.norm(yk - y_proj)
        while err > self.tol and k < self.max_iter:
            rho = self._update_rho(func, yk, rho)
            Fk_val = func(yk)
            yk1 = self.proj.project(yk, yk - rho * Fk_val, rho)
            Fk_val_1 = func(yk1)
            err = np.linalg.norm(yk1 - self.proj.project(yk1, yk1 - rho * Fk_val_1, rho))
            yk = yk1
            k += 1
        success = (err <= self.tol)
        return (yk, func(yk), err, success, k)

    def _update_rho(self, func, yk, rho):
        Fk_val = func(yk)
        yk1 = self.proj.project(yk, yk - rho * Fk_val, rho)
        rk = self._get_rk(func, yk1, yk, rho)
        while rk > self.L:
            rho = self.nu * rho
            yk1 = self.proj.project(yk, yk - rho * Fk_val, rho)
            rk = self._get_rk(func, yk1, yk, rho)
        if rk <= self.Lmin:
            rho = (1.0 / self.nu) * rho
        return rho

    def _get_rk(self, func, yk1, yk, rho):
        num = rho * np.linalg.norm(func(yk1) - func(yk))
        den = np.linalg.norm(yk1 - yk)
        return 0.0 if den == 0.0 else (num / den)

    def _numerical_jacobian(self, func, y, eps=1e-8):
        n = len(y)
        J = np.zeros((n, n), dtype=y.dtype)
        F0 = func(y)
        for i in range(n):
            y_eps = y.copy()
            y_eps[i] += eps
            F_eps = func(y_eps)
            J[:, i] = (F_eps - F0) / eps
        return J
