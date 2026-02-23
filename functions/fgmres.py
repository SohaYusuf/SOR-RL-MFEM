import numpy as np
from typing import Optional, Tuple, List
from scipy.linalg import get_blas_funcs, solve_triangular
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator

# prefer explicit module imports to avoid wildcard/circular-import issues
from functions.preconditioners import M_sor


# this code works on 09/16/2025 2:37 PM
class FlexibleGMRES_original:
    def __init__(self, A, max_iter, tol, M=None, omega = None):
        self.A = A # use A as linear operator
        # self.M = M if M is not None else np.eye(A.shape[0])  # Use identity matrix if no preconditioner is given
        self.M = M
        self.max_iter = max_iter
        self.tol = tol
        self.omega = omega

    def solve(self, b, x0=None):
        # self.A,self.M,x,b,postprocess = make_system(self.A,self.M,x0,b)
        if x0 is None:
            x0 = np.zeros_like(b)  # Initialize x0 as an array of zeros if not provided
            
        r0 = b - self.A @ x0
        beta = np.linalg.norm(r0)
        v = np.zeros((len(b), self.max_iter + 1))
        v[:, 0] = r0 / beta
        H = np.zeros((self.max_iter + 1, self.max_iter))
        Z = np.zeros((self.A.shape[0], self.max_iter))
        residuals = []

        print(f'\n self.M is : {self.M} \n')
        print(f'\n self.omega is : {self.omega} \n')

        for j in range(self.max_iter):
            # Step 3: Compute z_j
            # check convergence 
            # Step 3: Compute z_j with the correct handling of M
            if self.M is None:  # If M is None or a matrix
                #z_j = np.linalg.solve(self.M, v[:, j].reshape(-1, 1)).flatten()  # Ensure it's 2D for solve
                z_j = v[:,j] 
            elif isinstance(self.M, np.ndarray):
                z_j = np.linalg.solve(self.M, v[:, j].reshape(-1, 1)).flatten()
                # z_j = self.omega * z_j
            elif callable(self.M):  # If M is a linear operator (function)
                z_j = self.M(v[:, j])  # Apply the linear operator directly
                # z_j = self.omega * z_j
            else:
                raise ValueError("Preconditioner M must be either None, a matrix, or a callable linear operator.")
            
            Z[:, j] = z_j

            # Step 4: Compute w
            w = self.A @ z_j

            # Steps 5-8: Arnoldi iteration
            for i in range(0, j+1): # check if this is j
                H[i, j] = np.dot(w, v[:, i])
                w[:] -= H[i, j] * v[:, i]

            # Step 9: Compute h_{j+1,j} and v_{j+1}
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < self.tol:
                break
            v[:, j + 1] = w / H[j + 1, j]

            # Calculate and store residual
            #  print(f'H[:j + 1, :j + 1]: {H[:j + 1, :j + 1]}')
            # print(f'shape of H[:j + 1, :j]: {H[:j + 2, :j+1].shape}')
            y_approx = np.linalg.lstsq(H[:j + 2, :j+1], beta * np.eye(j + 2, 1).ravel(), rcond=None)[0]
            x_approx = x0 + Z[:, :j + 1] @ y_approx
            residuals.append(np.linalg.norm(b - self.A @ x_approx))

            # Check if the residual is below the tolerance
            if np.linalg.norm(b - self.A @ x_approx) <= self.tol:
                break

        # Final solution
        # use y_approx
        y_final = np.linalg.lstsq(H[:j + 1, :j + 1], beta * np.eye(j + 1, 1).ravel(), rcond=None)[0]
        x_m = x0 + Z[:, :j + 1] @ y_final

        return x_m, residuals




class FlexibleGMRES_RL:

    def __init__(self, A, max_iter, tol, M=None):
        self.A = A # use A as linear operator
        # self.M = M if M is not None else np.eye(A.shape[0])  # Use identity matrix if no preconditioner is given
        self.max_iter = max_iter
        self.tol = tol
        self.j = 0

    def initialize(self, b, x0=None):

        self.b = b
        if x0 is None:
            x0 = np.zeros_like(self.b)  # Initialize x0 as an array of zeros if not provided

        r0 = self.b - self.A @ x0
        self.beta = np.linalg.norm(r0)
        self.v = np.zeros((len(self.b), self.max_iter + 1))
        self.v[:, 0] = r0 / self.beta
        self.H = np.zeros((self.max_iter + 1, self.max_iter))
        self.Z = np.zeros((self.A.shape[0], self.max_iter))
        self.x0 = x0
        self.j = 0
        self.residuals = []
        self.omega = 0.0

    def step(self, M=None, omega=None):
        if self.j >= self.max_iter:
            raise ValueError(f"Maximum iterations {self.j} reached. Cannot perform another step.")

        # --- apply preconditioner robustly ---
        vj = self.v[:, self.j]
        if M is None:
            z_j = vj.copy()
        elif isinstance(M, np.ndarray):
            z_j = np.linalg.solve(M, vj.reshape(-1, 1)).ravel()
        elif isinstance(M, LinearOperator):
            z_j = M.matvec(vj)
        elif hasattr(M, 'matvec'):
            z_j = M.matvec(vj)
        elif callable(M):
            # last resort: call it
            z_j = M(vj)
        else:
            raise ValueError("Preconditioner M must be None, ndarray, LinearOperator, or callable.")

        # store Z
        self.Z[:, self.j] = z_j

        # Compute w = A * z_j
        w = self.A @ z_j

        # Arnoldi: orthogonalize against previous v's
        for i in range(0, self.j + 1):
            hij = np.dot(w, self.v[:, i])
            self.H[i, self.j] = hij
            w -= hij * self.v[:, i]

        # compute next subdiagonal entry and check for breakdown
        h = np.linalg.norm(w)
        self.H[self.j + 1, self.j] = h

        eps = 1e-16
        if h <= eps:
            # Breakdown: cannot form next basis vector. compute approx solution and return.
            try:
                rhs = self.beta * np.eye(self.j + 2, 1).ravel()
                y_approx = np.linalg.lstsq(self.H[:self.j + 2, :self.j+1], rhs, rcond=None)[0]
            except Exception:
                # fallback
                Hmat = self.H[:self.j + 2, :self.j+1].astype(np.float64, copy=False)
                rhs = (self.beta * np.eye(self.j + 2, 1).ravel()).astype(np.float64)
                y_approx = np.linalg.pinv(Hmat) @ rhs
            x_approx = self.x0 + self.Z[:, :self.j+1] @ y_approx
            residual_vector = self.b - self.A @ x_approx
            residual_norm = np.linalg.norm(residual_vector)
            self.residuals.append(residual_norm)
            # Do not increment j further; indicate convergence/breakdown
            return False, residual_vector, x_approx, residual_norm, self.residuals

        # Normal continuation
        self.v[:, self.j + 1] = w / h

        # Compute current least-squares solution y (robustly)
        try:
            rhs = self.beta * np.eye(self.j + 2, 1).ravel()
            y_approx = np.linalg.lstsq(self.H[:self.j + 2, :self.j+1], rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            # fallback to scipy (gelsy) or pinv
            try:
                y_approx, *_ = sp.linalg.lstsq(self.H[:self.j + 2, :self.j+1], rhs)
            except Exception:
                Hmat = self.H[:self.j + 2, :self.j+1].astype(np.float64, copy=False)
                y_approx = np.linalg.pinv(Hmat) @ rhs

        x_approx = self.x0 + self.Z[:, :self.j + 1] @ y_approx
        residual_vector = self.b - self.A @ x_approx
        residual_norm = np.linalg.norm(residual_vector)
        # store residual
        self.residuals.append(residual_norm)

        # increment j AFTER we computed everything for current j
        self.j += 1

        # check small subdiagonal of newly computed column (use local h saved earlier)
        if h < self.tol:
            return False, residual_vector, x_approx, residual_norm, self.residuals

        return True, residual_vector, x_approx, residual_norm, self.residuals