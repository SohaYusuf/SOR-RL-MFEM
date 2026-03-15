import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import eigsh

from functions.read_data_advection import read_data

def solve_advection_diffusion(config):
    """
    Solve du/dt = L(t) u,  L(t) = -c(t)*A_c + mu*A_d
    with the trapezoidal rule:
      (I - dt/2 L^{n+1}) u^{n+1} = (I + dt/2 L^n) u^n
    """

    # read input matrices/vectors
    data, _ = read_data(config['train_data_path'])
    # A_c = data['Advection']['A_c']
    # A_d = data['Diffusion']['A_d']
    # K = data['Advection']['K']
    # M = data['Advection']['M']

    # read stored matrices
    K = data['Advection']['K']           # advection matrix K (sparse)
    M = data['Advection']['M']           # mass matrix M (sparse)
    S = data['Diffusion']['A_d']         # diffusion / stiffness matrix (called A_d or S)

    A_c, A_d = apply_M_inverse(S, M, K)

    # numerical params
    dt = float(config.get('dt', 1e-4))
    tf = float(config.get('tf', 0.1))
    mu = float(config.get('mu', 0.1))
    # c = config.get('c', (lambda t: 1.0 + 0.5 * np.sin(2 * np.pi * t)))  # time-dependent default
    c = config.get('c', 1.0)  # time-dependent default
    save_every = int(config.get('save_every', 1))
    u0_type = config.get('u0_type', None)
    u0 = None
    # --- determine size first so we can build analytic u0 ---
    n = A_c.shape[0]

    # fallback zero if no vector provided
    if u0 is None:
        u0 = np.zeros(n, dtype=float)
    else:
        u0 = np.ravel(u0).astype(float)

    # Use analytic initial condition u_0(x,y)=sin(pi x) sin(pi y) when n is a perfect square.
    # We place nodes at cell centers (x_i = (i+0.5)/m) on [0,1]^2.
    m = int(np.round(np.sqrt(n)))
    if m * m == n:
        xs = (np.arange(m) + 0.5) / m
        ys = (np.arange(m) + 0.5) / m
        X, Y = np.meshgrid(xs, ys)
        # given f_func(x,y,t)
        # f_n = f_func(X.ravel(), Y.ravel(), t_n)      # length-n array
        # f_np1 = f_func(X.ravel(), Y.ravel(), t_np1)
        u = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel())
        u0 = u.astype(float)
        u0_analytic = u.copy() 
        print(f"Using analytic u0 on {m}x{m} grid for n={n}")
    else:
        print(f"n={n} is not a perfect square; keeping original u0 (or zeros)")

    # --- print parameters and matrix information ---
    print("\n===== Simulation Parameters =====")
    print(f"dt = {dt}")
    print(f"tf = {tf}")
    print(f"mu = {mu}")
    print(f"c  = {c}")
    print(f"u0 shape = {u0.shape}")
    print(f"A_c shape = {A_c.shape}, nnz = {A_c.nnz if hasattr(A_c,'nnz') else 'N/A'}")
    print(f"A_d shape = {A_d.shape if A_d is not None else 'None'}", 
        f", nnz = {A_d.nnz if (A_d is not None and hasattr(A_d,'nnz')) else 'N/A'}")
    print("=================================\n")

    def trapezoidal_method(mu, dt, tf, c, u, save_every=1):
    
        I = sp.identity(n, format='csc')

        # helper to build L from scalar c_val
        def build_L(c_val):
            A_diff = A_d if A_d is not None else sp.csc_matrix((n, n))
            return (float(c_val) * A_c - mu * A_diff).tocsc()
            # return (-float(c_val) * A_c + mu * A_diff).tocsc()

        # time stepping setup
        nt = int(np.ceil(tf / dt))
        times = np.linspace(0.0, nt * dt, nt + 1)
        snapshots = []
        c_is_callable = callable(c)

        if not c_is_callable:
            c0 = float(c)
            L_const = build_L(c0)
            left_const = (I - 0.5 * dt * L_const).tocsc()
            right_const = (I + 0.5 * dt * L_const).tocsc()
            LU_const = splu(left_const)
            for k, t in enumerate(times[:-1]):
                rhs = right_const.dot(u)
                u = LU_const.solve(rhs)
                # evaluate source vectors f_n, f_np1 as length-n numpy arrays
                # rhs = right.dot(u) + 0.5*dt*(f_n + f_np1)
                # u = LU.solve(rhs)
                if k % save_every == 0:
                    snapshots.append(u.copy())
        else:
            for k, t in enumerate(times[:-1]):
                t_n = t
                t_np1 = t + dt
                c_n = float(c(t_n))
                c_np1 = float(c(t_np1))
                L_n = build_L(c_n)
                L_np1 = build_L(c_np1)
                left = (I - 0.5 * dt * L_np1).tocsc()
                right = (I + 0.5 * dt * L_n).tocsc()
                LU = splu(left)
                rhs = right.dot(u)
                u = LU.solve(rhs)
                # evaluate source vectors f_n, f_np1 as length-n numpy arrays
                # rhs = right.dot(u) + 0.5*dt*(f_n + f_np1)
                # u = LU.solve(rhs)
                if k % save_every == 0:
                    snapshots.append(u.copy())

        snapshots.append(u.copy())

        return {'times': times, 'solutions': snapshots, 'u_final': u}

    
    if config['pure_advection_test']==1:

        print('Running pure advection problem ......')
        result = trapezoidal_method(mu=0.0, dt=dt, tf=tf, c=c, u=u, 
                                save_every=save_every)
        u_ref = np.sin(np.pi * (X - c * tf)) * np.sin(np.pi * Y)
        u_ref = u_ref.ravel()
        u = result['u_final']
        solution_accuracy_test(u_ref, u, n)

    elif config['pure_diffusion_test']==1:

        print('Running pure diffusion problem ......')
        result = trapezoidal_method(mu=mu, dt=dt, tf=tf, c=0.0, u=u, 
                                save_every=save_every)
        factor = np.exp(-2.0 * np.pi**2 * mu * result['times'][-1])       # analytic decay factor
        u_ref = factor * u0_analytic
        u = result['u_final']
        solution_accuracy_test(u_ref, u, n)
    
    else:
        print('Running advection diffusion problem ......')
        
        # # K = stiffness matrix, M = mass matrix (scipy sparse)
        # k = 1  # compute first nontrivial eigenpair
        # vals, vecs = eigsh(K, k=k, M=M, which='SM')   # 'SM' = smallest magnitude
        # lambda1 = float(vals[0])
        # phi = vecs[:,0]
        # # mass-normalize phi so phi^T M phi = 1
        # norm2 = (phi @ (M.dot(phi)))**0.5
        # phi = phi / norm2
        # # analytic factor at time t
        # factor = np.exp(-mu * lambda1 * tf)
        # u_ref = factor * phi

        result = trapezoidal_method(mu=mu, dt=dt, tf=tf, c=c, u=u, 
                                save_every=save_every)
        u = result['u_final']
        # solution_accuracy_test(u_ref, u, n)

    # Plot final solution:
    m = int(np.round(np.sqrt(n)))
    if m * m == n:
        U = u.reshape((m, m))
        plt.figure()
        plt.imshow(U, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(label='u')
        plt.title(f'Final solution u(T) — 2D ({m}x{m})')
    else:
        # non-analytic / vector IC — keep 1D plot
        plt.figure()
        plt.plot(u)
        plt.title('Final solution u(T)')
        plt.grid(True)

    plt.savefig('u.png')
    plt.show()

    return result

def solution_accuracy_test(u_ref, u, n):
    L2_err = np.linalg.norm(u - u_ref) / np.sqrt(n)       # discrete L2 on unit square
    max_err = np.max(np.abs(u - u_ref))
    rmse = np.sqrt(np.mean((u - u_ref)**2))            # root mean squared error
    rel_rmse = rmse / (np.sqrt(np.mean(u_ref**2)) + 1e-16)
    print(f"L2-error = {L2_err:.6e}, max-abs-error = {max_err:.6e}")
    print(f"RMSE = {rmse:.6e}, max-abs-error = {max_err:.6e}")
    print(f"Relative RMSE = {rel_rmse:.6e}")
    n=int(np.round(np.sqrt(n)))
    U = u.reshape((n, n))
    plt.figure()
    plt.imshow(U, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    plt.colorbar(label='u')
    plt.title(f'Final solution u(T) — 2D ({n}x{n})')
    plt.savefig('solution.png', dpi=300)
    plt.show()
    return rmse

def apply_M_inverse(S, M, K):
    # convert M to CSC and factorize once for repeated solves
    M_csc = M.tocsc()
    M_lu = splu(M_csc)   # uses a direct sparse LU factorization

    # helper: compute M^{-1} * A by solving M X = A column-by-column,
    # returning a sparse CSC matrix. This avoids forming dense M^{-1}.
    def apply_minv_to(A):
        A = A.tocsc()
        n = A.shape[1]
        rows = []
        cols = []
        vals = []
        for j in range(n):
            # get j-th column of A as dense vector (sparse->dense for a single column)
            col = A.getcol(j).toarray().ravel()
            x = M_lu.solve(col)          # solve M x = col  -> x = M^{-1} * A[:,j]
            nz = np.nonzero(x)[0]
            if nz.size:
                rows.extend(nz.tolist())
                cols.extend([j]*nz.size)
                vals.extend(x[nz].tolist())
        return sp.coo_matrix((vals, (rows, cols)), shape=A.shape).tocsc()

    # form operator-form matrices A_c = M^{-1} K and A_d = M^{-1} S
    A_c = apply_minv_to(K)
    A_d = apply_minv_to(S)
    return A_c, A_d