import numpy as np
import torch

from scipy.sparse import csr_matrix, diags, eye, tril, triu
from scipy.sparse.linalg import (
    LinearOperator,
    lsqr,
    splu,
    spilu,
    spsolve,
    spsolve_triangular,
)

from pyamg.relaxation.relaxation import sor as pyamg_sor


def M_sor(A, omega=1.0, epsilon=1e-20):
    """
    Build an SOR preconditioner M^{-1} for A = D - E - F, i.e.
        M = (D - ω E)/ω
    so that
        M^{-1} x = ω (D - ω E)^{-1} x
    is applied via one sparse forward‑substitution.
    
    Parameters
    ----------
    A : (n,n) array or sparse matrix
        The system matrix (may be non‑symmetric).
    omega : float, optional
        Relaxation parameter (ω=1 is Gauss–Seidel).
    epsilon : float, optional
        Tiny shift to the diagonal of M to avoid zero pivots.
    
    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        LinearOperator that applies M^{-1} via matvec (and rmatvec).
    """
    # 1) Ensure CSR
    A = csr_matrix(A)
    n = A.shape[0]

    # 2) Extract D and E (with A = D - E - F)
    D = diags(A.diagonal(), format='csr')
    E = -tril(A, k=-1, format='csr')   # so that A = D - E - F

    # 3) Build M = (D - ω E)/ω  (still lower‑triangular + diag)
    M_mat = (D - omega * E) / omega

    # 4) Add tiny shift to diagonal, but keep sparse
    M_mat = M_mat + epsilon * diags([1.0], [0], shape=(n, n), format='csr')

    # 5) Define the action of M^{-1}: solve M_mat * y = x by forward substitution
    def _apply(x):
        # spsolve_triangular knows it's lower‑triangular
        return spsolve_triangular(M_mat, x, lower=True)

    # 6) Return a LinearOperator wrapping that solve
    return LinearOperator(
        matvec=_apply,
        rmatvec=_apply,      # same for left‑preconditioning
        shape=(n, n),
        dtype=A.dtype
    )