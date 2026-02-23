import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
import pdb

def sor(A, b, x=None, omega=1.0, tol=1e-10, max_iter=300):
    """
    Direct translation of the provided MATLAB SOR code.
    Returns (k, out, residuals) where residuals is a list of residual norms
    normalized by the initial residual (initial entry will be 1.0).
    """
    # if A is a sparse matrix with toarray, convert to dense like MATLAB would
    if hasattr(A, "toarray"):
        A = A.toarray()

    if x is None:
        x = np.zeros(A.shape[0])
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    D = np.diag(np.diag(A))
    M = D / omega + np.tril(A, -1)

    r = b - A @ x
    norm0 = np.linalg.norm(r)

    # Prepare residual history normalized by initial residual.
    if norm0 == 0:
        # initial residual is zero -> normalized history is [0.0], zero iterations
        return 0, x.copy(), [0.0]

    residuals = [1.0]  # norm(r)/norm0 for initial r is 1.0

    # cap iterations at 10000 as in the MATLAB code
    for k in range(1, max_iter+1):
        x = x + solve_triangular(M, r, lower=True, check_finite=300)
        r = b - A @ x
        res_norm = np.linalg.norm(r) / norm0
        residuals.append(res_norm)
        if res_norm < tol:
            out = x
            return k, out, residuals

    # reached cap without meeting tolerance
    out = x
    return k, out, residuals

# try to import the sparse triangular solver (newer SciPy); if missing, we'll fall back
try:
    from scipy.sparse.linalg import spsolve_triangular as _spsolve_triangular
    HAS_SPSOLVE_TRIANG = True
except Exception:
    from scipy.sparse.linalg import spsolve as _spsolve_fallback
    HAS_SPSOLVE_TRIANG = False

def sor_sparse(A, b, x=None, omega=1.0, tol=1e-6, max_iter=1000):
    """
    SOR (MATLAB-style) that supports both dense numpy arrays and scipy sparse matrices.
    Returns (k, x_out, residuals) where residuals are normalized by initial residual.
    """
    is_sparse = sparse.isspmatrix(A)
    if is_sparse:
        A_sp = A.tocsr()
        n = A_sp.shape[0]
    else:
        A = np.asarray(A, dtype=float)
        n = A.shape[0]

    b = np.asarray(b, dtype=float).reshape(-1)
    if x is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float).reshape(-1)

    # build lower-triangular M = D/omega + tril(A,-1)
    if is_sparse:
        diag = A_sp.diagonal()
        D_over_omega = sparse.diags(diag / omega, format='csr')
        L = sparse.tril(A_sp, k=-1, format='csr')
        M = D_over_omega + L  # sparse lower triangular
    else:
        diag = np.diag(A)
        M = np.diag(diag / omega) + np.tril(A, -1)  # dense lower triangular

    # initial residual and normalization
    r = (b - A_sp.dot(x)) if is_sparse else (b - A @ x)
    norm0 = float(np.linalg.norm(r))
    if norm0 == 0.0:
        return 0, x.copy(), [0.0]

    residuals = [1.0]
    for k in range(1, max_iter + 1):
        # solve M * delta = r for lower-triangular M
        if is_sparse:
            if HAS_SPSOLVE_TRIANG:
                delta = _spsolve_triangular(M, r, lower=True)
            else:
                # fallback to spsolve (works but may be slower)
                delta = _spsolve_fallback(M.tocsr(), r)
        else:
            delta = solve_triangular(M, r, lower=True, check_finite=False)

        x = x + np.asarray(delta).reshape(-1)
        r = (b - A_sp.dot(x)) if is_sparse else (b - A @ x)
        res_norm = float(np.linalg.norm(r)) / norm0
        residuals.append(res_norm)
        if res_norm < tol:
            return k, x.copy(), residuals

    return k, x.copy(), residuals