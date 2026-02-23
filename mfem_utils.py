# mfem_utils.py
import numpy as np
import mfem.ser as mfem

class PyJacobiPreconditioner(mfem.Solver):
    """
    A lightweight Jacobi / relaxed-Jacobi preconditioner implemented as
    a Python subclass of mfem.Solver so it can be passed to SetPreconditioner().

    Usage:
        prec = PyJacobiPreconditioner(omega=1.0)
        prec.SetOperator(A)           # A is an mfem.SparseMatrix
        solver.SetPreconditioner(prec)
    """

    def __init__(self, omega=1.0):
        # Initialize base Solver (no size argument required here)
        # Note: depending on your MFEM build you may optionally pass a size.
        mfem.Solver.__init__(self)
        self.A = None
        self.diag = None
        self.n = 0
        self.omega = float(omega)

    def SetOperator(self, A):
        """Store matrix A (mfem.SparseMatrix) and extract diagonal robustly."""
        self.A = A
        n = A.Height()
        self.n = n
        self.diag = np.zeros(n, dtype=float)

        # Try GetDiag(i) if present (fast)
        got_diag = True
        try:
            for i in range(n):
                self.diag[i] = A.GetDiag(i)
        except Exception:
            got_diag = False

        if not got_diag:
            # Try A.GetRow(i) if present; otherwise fallback to A.Mult unit vector
            for i in range(n):
                found = False
                try:
                    # many SWIG wrappers expose GetRow(i) returning (cols, vals)
                    cols, vals = A.GetRow(i)
                    for c, v in zip(cols, vals):
                        if c == i:
                            self.diag[i] = v
                            found = True
                            break
                except Exception:
                    found = False

                if not found:
                    # fallback: compute A * e_i and read entry i
                    ei = mfem.Vector(n)
                    ei.Assign(0.0)
                    ei[i] = 1.0
                    out = mfem.Vector(n)
                    out.Assign(0.0)
                    A.Mult(ei, out)
                    self.diag[i] = out[i]

        # Guard against zero diagonal
        self.diag[self.diag == 0.0] = 1.0

    def SetOmega(self, omega):
        self.omega = float(omega)

    def Mult(self, x, y):
        """
        Apply preconditioner: y = omega * D^{-1} x
        x, y are mfem.Vector objects.
        """
        # print(self.omega)
        n = self.n
        if n == 0 or self.diag is None:
            # If operator wasn't set, act as identity (safe fallback)
            for i in range(x.Size()):
                y[i] = x[i]
            return

        # Fast path: try to access underlying numpy buffer
        try:
            xb = x.GetDataArray()    # may return a numpy view
            yb = y.GetDataArray()
            # element-wise application
            # avoid creating temporaries in Python loops when possible
            for i in range(n):
                yb[i] = self.omega * xb[i] / self.diag[i]
            return
        except Exception:
            # Fallback: element-wise access via [] operator
            for i in range(n):
                y[i] = self.omega * x[i] / self.diag[i]
            return



# import mfem.ser as mfem
# import numpy as np

# class SimpleJacobiPreconditioner:
#     """
#     Very simple relaxed Jacobi preconditioner:

#         y = omega * D^{-1} x

#     where D is the diagonal of the matrix A.
#     """

#     def __init__(self, omega=1.0):
#         self.A = None
#         self.diag = None
#         self.omega = omega

#     def SetOperator(self, A):
#         """
#         Called by the solver to give the matrix.
#         We extract and store the diagonal.
#         """
#         self.A = A
#         n = A.Height()
#         self.diag = np.zeros(n)

#         # Extract diagonal entries
#         for i in range(n):
#             self.diag[i] = A.GetDiag(i)

#         # Avoid division by zero
#         self.diag[self.diag == 0.0] = 1.0

#     def SetOmega(self, omega):
#         """
#         Allow RL agent to change relaxation parameter.
#         """
#         self.omega = float(omega)

#     def Mult(self, x, y):
#         """
#         Apply preconditioner: y = omega * D^{-1} x
#         """
#         for i in range(len(self.diag)):
#             y[i] = self.omega * x[i] / self.diag[i]
