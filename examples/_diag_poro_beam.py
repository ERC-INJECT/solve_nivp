"""Diagnostic: inspect poro descriptor (A, rhs_jac) at the static state."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from poro_beam_oscillation_comparison import build_poro_beam_system

ps = build_poro_beam_system()
dyn = ps["dyn"]
y0 = ps["y0"]
n_base = ps["n_base"]

A = dyn["A"]
rhs = dyn["rhs"]
rhs_jac = dyn["rhs_jac"]

if sp.issparse(A):
    A_dense = A.toarray()
else:
    A_dense = np.asarray(A)

print("y0 shape:", y0.shape, " n_base=", n_base, " Np=", ps["Np"], " Nu=", ps["Nu"])
print("A shape:", A_dense.shape, " nnz-like:", int(np.count_nonzero(A_dense)))
print("A diag nonzero count:", int(np.count_nonzero(np.diag(A_dense))))

# component slices
cs = dyn.get("component_slices", None)
print("component_slices:", cs)

# residual at static state
r0 = rhs(0.0, y0)
print("||rhs(0, y_static)||_inf:", float(np.max(np.abs(r0))))
print("rhs[:n_base] max:", float(np.max(np.abs(r0[:n_base]))))
print("rhs[n_base:] max:", float(np.max(np.abs(r0[n_base:]))))

# Jacobian structure
J = rhs_jac(0.0, y0)
if sp.issparse(J):
    J = J.tocsr()
print("J shape:", J.shape, " nnz:", J.nnz)

# Pencil eigenvalues (A y' = J y) -- compute a few
# Reduce to a smaller system: take indices where A has nonzero row
row_nz = np.where(np.abs(A_dense).sum(axis=1) > 0)[0]
print("rows of A with nonzero entries:", row_nz.size, "/", A_dense.shape[0])

# Block structure: how is A partitioned?
print("A[:n_base, :n_base] max:", float(np.max(np.abs(A_dense[:n_base, :n_base]))))
print("A[:n_base, n_base:] max:", float(np.max(np.abs(A_dense[:n_base, n_base:]))))
print("A[n_base:, :n_base] max:", float(np.max(np.abs(A_dense[n_base:, :n_base]))))
print("A[n_base:, n_base:] max:", float(np.max(np.abs(A_dense[n_base:, n_base:]))))

# velocity block — what does rhs_jac say about d(y')/dy across the pencil?
# For oscillation we need J to have purely imaginary eigenvalues.
try:
    # Dense generalized eigenvalue on a reasonable-size subsystem
    if A_dense.shape[0] <= 1500:
        from scipy.linalg import eig
        w, _ = eig(J.toarray(), A_dense)
        finite = w[np.isfinite(w)]
        print("finite eig count:", finite.size)
        print("max |Re(eig)|:", float(np.max(np.abs(finite.real))) if finite.size else None)
        print("max |Im(eig)|:", float(np.max(np.abs(finite.imag))) if finite.size else None)
        imag_nonzero = finite[np.abs(finite.imag) > 1e-6]
        print("# eigenvalues with nonzero Im:", imag_nonzero.size)
        if imag_nonzero.size:
            top_im = np.sort(np.abs(imag_nonzero.imag))[-5:]
            print("top 5 |Im(eig)|:", top_im)
except Exception as e:
    print("eig failed:", e)
