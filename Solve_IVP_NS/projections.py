import numpy as np
from abc import ABC, abstractmethod
import warnings

# Optional: Try importing JAX and autograd for Jacobian acceleration.
try:
    import jax
    from jax.config import config
    # Enable 64-bit in JAX if desired:
    config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import Array as JaxArray
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    JaxArray = ()

try:
    import autograd
    from autograd import jacobian as ag_jacobian
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False


##############################################################################
# Utility functions for JAX/NumPy compatibility
##############################################################################
def _is_jax_array(x):
    """Returns True if x is a JAX tracer or JAX array."""
    return (JAX_AVAILABLE and isinstance(x, JaxArray))

def _safe_at_set(arr, idx, val):
    """
    If arr is a NumPy array => in-place assignment arr[idx] = val
    If arr is a JAX array => arr = arr.at[idx].set(val)
    Returns the updated arr.
    """
    if _is_jax_array(arr):
        return arr.at[idx].set(val)
    else:
        arr[idx] = val
        return arr

def _safe_at_set_vector(arr, idxs, vals):
    """
    Like _safe_at_set but for multiple indices at once.
    If arr is JAX => do a small loop or arr = arr.at[idx].set(...) for each item.
    If arr is NumPy => arr[idxs] = vals.
    """
    if _is_jax_array(arr):
        # if idxs, vals are arrays => do a short loop
        for i, index_val in enumerate(idxs):
            arr = arr.at[index_val].set(vals[i])
        return arr
    else:
        arr[idxs] = vals
        return arr

def _to_numpy_if_needed(x):
    """
    If x is a JAX array, convert it to NumPy. Otherwise return x as is.
    Helpful for numerical jacobian.
    """
    if _is_jax_array(x):
        return np.array(x)
    return x


##############################################################################
# Base Projection
##############################################################################
class Projection(ABC):
    def __init__(self, component_slices=None):
        """
        component_slices is an optional list of slices for sub-portions
        of the full state array.
        """
        self.component_slices = component_slices if component_slices is not None else []

    @abstractmethod
    def project(self, current_state, candidate, rhok=None):
        pass

    @abstractmethod
    def tangent_cone(self, candidate, current_state, rhok=None):
        pass


##############################################################################
# IdentityProjection
##############################################################################
class IdentityProjection(Projection):
    def project(self, current_state, candidate, rhok=None):
        return candidate

    def tangent_cone(self, candidate, current_state, rhok=None):
        n = candidate.shape[0]
        return np.eye(n)


##############################################################################
# SignProjection
##############################################################################
class SignProjection(Projection):
    """
    Orthogonal projection onto the set:
      K = { (y,w) :  w in sign(y) }.
    i.e. y>0 => w=+1, y<0 => w=-1, y=0 => w in [-1,1].
    Only the w-components are modified; the y-components remain unchanged.
    """
    def __init__(self, y_indices, w_indices, component_slices=None):
        super().__init__(component_slices=component_slices)
        self.y_indices = np.array(y_indices) if not np.isscalar(y_indices) else y_indices
        self.w_indices = np.array(w_indices) if not np.isscalar(w_indices) else w_indices

    def project(self, current_state, candidate, rhok=None):
        """
        If y>0 => w=+1, y<0 => w=-1, else clip w to [-1,1].
        """
        new_candidate = candidate
        # Let's separate code for JAX vs NumPy
        if _is_jax_array(candidate):
            # JAX path
            y = new_candidate[self.y_indices]
            w = new_candidate[self.w_indices]
            if y.shape == ():  # scalar case
                # use jnp.where to handle sign logic
                w_new = jnp.where(y>0, 1.0, jnp.where(y<0, -1.0, jnp.clip(w, -1.0, 1.0)))
            else:
                # vector
                w_new = jnp.where(
                    y>0, 1.0,
                    jnp.where(y<0, -1.0, jnp.clip(w, -1.0, 1.0))
                )
            # Now set those in new_candidate
            if isinstance(self.w_indices, np.ndarray) and self.w_indices.size>1:
                new_candidate = _safe_at_set_vector(new_candidate, self.w_indices, w_new)
            else:
                new_candidate = _safe_at_set(new_candidate, self.w_indices, w_new)
        else:
            # NumPy path
            new_candidate = new_candidate.copy()
            y = new_candidate[self.y_indices]
            w = new_candidate[self.w_indices]
            if np.isscalar(y):
                new_candidate[self.w_indices] = 1.0 if y>0 else (-1.0 if y<0 else np.clip(w, -1.0, 1.0))
            else:
                new_candidate[self.w_indices] = np.where(
                    y>0, 1.0,
                    np.where(y<0, -1.0, np.clip(w, -1.0, 1.0))
                )
        return new_candidate

    def tangent_cone(self, candidate, current_state, rhok=None):
        """
        Derivative: if y!=0 => w is constant => derivative=0
        if y==0 => derivative=1 if w in(-1,1), else0
        """
        n = candidate.shape[0]
        D = np.eye(n)
        y = np.atleast_1d(current_state[self.y_indices])
        w = np.atleast_1d(current_state[self.w_indices])
        mask_nonzero = (y!=0)
        mask_zero    = (y==0)

        if not np.isscalar(self.w_indices):
            w_idx = self.w_indices
        else:
            w_idx = np.array([self.w_indices])

        # For y!=0 => derivative=0
        D[w_idx[mask_nonzero], w_idx[mask_nonzero]] = 0.0
        # For y==0 => derivative=1 if w in(-1,1)
        D[w_idx[mask_zero], w_idx[mask_zero]] = np.where(
            (w[mask_zero]>-1)&(w[mask_zero]<1), 1.0,0.0
        )
        return D


##############################################################################
# CoulombProjection
##############################################################################
class CoulombProjection(Projection):
    def __init__(self, con_force_func, rhok, component_slices=None, constraint_indices=None, jac_mode='auto'):
        super().__init__(component_slices)
        self.con_force_func = con_force_func
        self.rhok = rhok
        self.jac_mode = jac_mode.lower()
        self._autodiff_failed = False

        if constraint_indices is not None:
            self.constraint_indices = np.array(constraint_indices)
        else:
            if self.component_slices:
                self.constraint_indices = np.concatenate(
                    [np.arange(sl.start, sl.stop) for sl in self.component_slices]
                )
            else:
                self.constraint_indices = np.array([])

        self._setup_jacobian()

    def _setup_jacobian(self):
        """Try autodiff if requested, else numerical."""
        self.jac_func = None
        if self.jac_mode == 'auto':
            if JAX_AVAILABLE:
                try:
                    jax_func = jax.jit(lambda y: self.con_force_func(y))
                    self.jac_func = jax.jit(jax.jacfwd(jax_func))
                    self.jac_mode = 'jax'
                except Exception as e:
                    warnings.warn(f"JAX autodiff failed: {e}. Falling back to numerical differentiation.")
                    self.jac_mode = 'numerical'
            elif AUTOGRAD_AVAILABLE:
                try:
                    self.jac_func = ag_jacobian(self.con_force_func)
                    self.jac_mode = 'autograd'
                except Exception as e:
                    warnings.warn(f"Autograd autodiff failed: {e}. Falling back to numerical differentiation.")
                    self.jac_mode = 'numerical'
            else:
                self.jac_mode = 'numerical'
        elif self.jac_mode == 'jax' and JAX_AVAILABLE:
            try:
                jax_func = jax.jit(lambda y: self.con_force_func(y))
                self.jac_func = jax.jit(jax.jacfwd(jax_func))
            except Exception as e:
                warnings.warn(f"JAX autodiff failed: {e}. Falling back to numerical differentiation.")
                self.jac_mode = 'numerical'
        elif self.jac_mode == 'autograd' and AUTOGRAD_AVAILABLE:
            try:
                self.jac_func = ag_jacobian(self.con_force_func)
            except Exception as e:
                warnings.warn(f"Autograd autodiff failed: {e}. Falling back to numerical differentiation.")
                self.jac_mode = 'numerical'

    def _compute_jacobian(self, y):
        """
        If we haven't failed autodiff => try it. Otherwise do numerical.
        """
        if self._autodiff_failed:
            return self._numerical_jacobian(y)

        if self.jac_mode in ['jax','autograd'] and self.jac_func is not None:
            try:
                jac = self.jac_func(y)
                if not isinstance(jac, np.ndarray):
                    jac = np.array(jac)
                return jac
            except Exception as e:
                warnings.warn(f"Autodiff Jacobian evaluation failed: {e}. Using numerical from now on.")
                self._autodiff_failed = True
                self.jac_mode = 'numerical'
                return self._numerical_jacobian(y)

        return self._numerical_jacobian(y)

    def _numerical_jacobian(self, y, eps=1e-8):
        y_np = _to_numpy_if_needed(y)
        n = len(y_np)
        J = np.zeros((n,n), dtype=y_np.dtype)
        f0 = _to_numpy_if_needed(self.con_force_func(y))
        for j in range(n):
            y_pert = y_np.copy()
            y_pert[j] += eps
            f_eps = _to_numpy_if_needed(self.con_force_func(y_pert))
            J[:, j] = (f_eps - f0)/eps
        return J

    @staticmethod
    def _projD_optimized(pts, friction_vals):
        """
        region-based projection onto friction cone { (v,z) : z >= |v| }
        pts shape=(N,2), friction_vals shape=(N,).
        """
        n1 = np.array([1,1])/np.sqrt(2)
        n2 = np.array([-1,1])/np.sqrt(2)
        d = np.zeros_like(pts)

        mask_con0 = (friction_vals==0)
        mask_non0 = ~mask_con0
        d[mask_con0] = pts[mask_con0]
        pts_non0 = pts[mask_non0]

        R1 = np.abs(pts_non0[:,1])<=pts_non0[:,0]
        R2 = np.abs(pts_non0[:,0])<=-pts_non0[:,1]
        R3 = np.abs(pts_non0[:,1])<=-pts_non0[:,0]

        d_proj = np.zeros_like(pts_non0)
        d_proj[R1] = (np.dot(pts_non0[R1], n1)[:,None])*n1
        d_proj[R3] = (np.dot(pts_non0[R3], n2)[:,None])*n2
        mask_no_proj = ~(R1|R2|R3)
        d_proj[mask_no_proj] = pts_non0[mask_no_proj]
        d[mask_non0] = d_proj
        return d

    @staticmethod
    def _projD(y, con_force_func, state, rhok, constraint_indices):
        """
        The main friction projection:
         second_column[ci] = state[ci+1] - rhok[ci]*conf[ci],
         pr_2d[ci+1,0] = pr_2d[ci,1].
        Must do it JAX-safely.
        """
        conf = con_force_func(state)
        second_column = conf.copy()

        ci = np.asarray(constraint_indices)
        if ci.size>0:
            # newvals = state[ci+1] - rhok[ci]*conf[ci]
            st_ci_plus = _to_numpy_if_needed(state[ci+1])
            rhok_ci = _to_numpy_if_needed(rhok[ci])
            conf_ci = _to_numpy_if_needed(conf[ci])
            newvals = st_ci_plus - (rhok_ci*conf_ci)

            # We'll do a JAX-safe assignment if needed:
            if _is_jax_array(second_column):
                if ci.size==1:
                    second_column = _safe_at_set(second_column, ci[0], newvals)
                else:
                    second_column = _safe_at_set_vector(second_column, ci, newvals)
            else:
                second_column[ci] = newvals

        pts = np.column_stack((y, second_column))
        pr_2d = CoulombProjection._projD_optimized(pts, conf)

        # now pr_2d[ ci+1, 0 ] = pr_2d[ ci,1 ]
        if ci.size>0:
            newvals2 = pr_2d[ci,1]
            if _is_jax_array(pr_2d):
                for i, cval in enumerate(ci):
                    pr_2d = pr_2d.at[cval+1,0].set(newvals2[i])
            else:
                pr_2d[ci+1,0] = newvals2
        return pr_2d[:,0]

    def project(self, current_state, candidate, rhok):
        n = candidate.shape[0]
        # build big_rhok
        if rhok is None:
            if _is_jax_array(candidate):
                big_rhok = jnp.ones(n, dtype=candidate.dtype)
            else:
                big_rhok = np.ones(n, dtype=candidate.dtype)
        elif np.isscalar(rhok):
            val = float(rhok)
            if _is_jax_array(candidate):
                big_rhok = jnp.full((n,), val, dtype=candidate.dtype)
            else:
                big_rhok = np.full((n,), val, dtype=candidate.dtype)
        else:
            # arraylike
            if _is_jax_array(candidate):
                big_rhok = jnp.zeros(n, dtype=candidate.dtype)
            else:
                big_rhok = np.zeros(n, dtype=candidate.dtype)
            # set each slice
            for i, sl in enumerate(self.component_slices):
                if hasattr(sl, 'start'):
                    # slice
                    for idx_ in range(sl.start, sl.stop):
                        big_rhok = _safe_at_set(big_rhok, idx_, float(rhok[i]))
                else:
                    # single index
                    big_rhok = _safe_at_set(big_rhok, sl, rhok[i])

        return CoulombProjection._projD(candidate, self.con_force_func, current_state, big_rhok, self.constraint_indices)

    def tangent_cone(self, candidate, current_state, rhok=None):
        """
        Build derivative matrix. We'll store in a NumPy array for the linear solve.
        """
        n = candidate.shape[0]
        D = np.eye(n)
        conf = self.con_force_func(current_state)
        J_conf = self._compute_jacobian(current_state)
        ci = np.asarray(self.constraint_indices)
        if ci.size==0:
            return D

        if rhok is None:
            rhok_full = np.ones(n, dtype=float)
        elif np.isscalar(rhok):
            rhok_full = np.full((n,), float(rhok), dtype=float)
        else:
            rhok_full = _to_numpy_if_needed(rhok)

        conf_np = _to_numpy_if_needed(conf)
        state_np = _to_numpy_if_needed(current_state)
        for idx in ci:
            z_idx = idx+1
            if z_idx>=n:
                continue
            v_proj = state_np[idx]
            z_proj = state_np[z_idx]
            dconf_dy = J_conf[idx]
            if z_proj>abs(v_proj):
                D[idx,idx]=1.0
                D[z_idx,z_idx]=1.0
            elif abs(z_proj-abs(v_proj))==0:
                dz_dy = -rhok_full[idx]*dconf_dy
                I = np.eye(n)
                if v_proj>0.0:
                    D[idx,:] =0.5*(I[idx,:]+dz_dy)
                    D[z_idx,:]=0.5*(I[idx,:]+dz_dy)
                elif v_proj<0.0:
                    D[idx,:] =0.5*(I[idx,:]-dz_dy)
                    D[z_idx,:]=0.5*(-I[idx,:]+dz_dy)
                else:
                    D[idx,idx]=0.0
                    D[z_idx,:]=0.0
            else:
                D[idx,idx]=1.0
                D[z_idx,z_idx]=1.0
        return D

