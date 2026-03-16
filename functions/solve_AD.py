import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix

from collections import namedtuple, deque
import random

# RL imports
import torch
import torch.optim as optim
from functions.env_fgmres import FMGRESEnv
from functions.fgmres import FlexibleGMRES_RL, FlexibleGMRES_original
from functions.model import DQN, optimize_model
from functions.read_data_advection import read_data

def solve_advection_diffusion(config):
    """
    Solve du/dt = L(t) u,  L(t) = -c(t)*A_c + mu*A_d
    with the trapezoidal rule:
      (I - dt/2 L^{n+1}) u^{n+1} = (I + dt/2 L^n) u^n

    Optional RL integration: set config['use_rl']=True and provide
    config['policy_checkpoint']='path/to/policy_net_weights.pth'.
    """

    # read input matrices/vectors
    data, _ = read_data(config['train_data_path'])

    # read stored matrices (keep original formats)
    K = data['Advection']['K']           # advection matrix K (sparse)
    M = data['Advection']['M']           # mass matrix M (sparse)
    S = data['Diffusion']['A_d']         # diffusion / stiffness matrix (called A_d or S)

    A_c, A_d = apply_M_inverse(S, M, K)

    # numerical params
    dt = float(config.get('dt', 1e-7))
    tf = float(config.get('tf', 1e-6))
    mu = float(config.get('mu', 0.1))
    c = config.get('c', (lambda t: 1.0 + 0.5 * np.sin(2 * np.pi * t)))  # time-dependent default

    save_every = int(config.get('save_every', 1))
    u0_type = config.get('u0_type', None)
    u0 = None
    # --- determine size first so we can build analytic u0 ---
    n = A_c.shape[0]
    nt = int(np.ceil(tf / dt))

    # fallback zero if no vector provided
    if u0 is None:
        u0 = np.zeros(n, dtype=float)
    else:
        u0 = np.ravel(u0).astype(float)

    # Use analytic initial condition u_0(x,y)=sin(pi x) sin(pi y) when n is a perfect square.
    m = int(np.round(np.sqrt(n)))
    if m * m == n:
        xs = (np.arange(m) + 0.5) / m
        ys = (np.arange(m) + 0.5) / m
        X, Y = np.meshgrid(xs, ys)
        u = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel())
        u0 = u.astype(float)
        u0_analytic = u.copy()
        print(f"Using analytic u0 on {m}x{m} grid for n={n}")
    else:
        print(f"n={n} is not a perfect square; keeping original u0 (or zeros)")

    # print parameters and matrix information
    print("\n===== Simulation Parameters =====")
    print(f"dt = {dt}")
    print(f"tf = {tf}")
    print("Number of time steps: ", nt)
    print(f"mu = {mu}")
    print(f"c  = {c}")
    print(f"u0 shape = {u0.shape}")
    print(f"A_c shape = {A_c.shape}, nnz = {A_c.nnz if hasattr(A_c,'nnz') else 'N/A'}")
    print(f"A_d shape = {A_d.shape if A_d is not None else 'None'}",
          f", nnz = {A_d.nnz if (A_d is not None and hasattr(A_d,'nnz')) else 'N/A'}")
    print("=================================\n")

    # RL runtime config
    use_rl = bool(config.get('use_rl', False))
    policy_checkpoint = config.get('policy_checkpoint', None)
    device = torch.device(config.get('device', 'cpu'))
    rl_max_iter = int(config.get('max_iter', 200))
    rl_tol = float(config.get('target_tol', 1e-8))
    # number of discrete actions (must match training)
    n_actions = int(config.get('n_actions', 11))

    def trapezoidal_method(mu, dt, tf, c, u, save_every=1):
        I = sp.identity(n, format='csc')

        # helper to build L from scalar c_val (operator-form used by code)
        def build_L(c_val):
            A_diff = A_d if A_d is not None else sp.csc_matrix((n, n))
            return (float(c_val) * A_c - mu * A_diff).tocsc()

        # prepare RL objects if requested (do once per trapezoidal run)
        if use_rl:
            env = FMGRESEnv(config=config)
            Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
            class ReplayMemory(object):
                def __init__(self, capacity):
                    self.memory = deque([], maxlen=capacity)
                def push(self, *args):
                    """Save a transition"""
                    self.memory.append(Transition(*args))
                def sample(self, batch_size):
                    return random.sample(self.memory, batch_size)
                def __len__(self):
                    return len(self.memory)
            BATCH_SIZE = config["batch_size"]
            GAMMA = config["gamma"]
            EPS_START = config["eps_start"]
            EPS_END = config["eps_end"]
            EPS_DECAY = config["eps_decay"]
            TAU = config["tau"]
            LR = config["learning_rate"]
            num_episodes = config["num_episodes"]
            n_actions = config['n_actions']
            n_observations = 5
            policy_net = DQN(n_observations, n_actions).to(device)
            target_net = DQN(n_observations, n_actions).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
            memory = ReplayMemory(10000)
            steps_done = 0
            def select_action(state):
                nonlocal steps_done
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1
                if sample > eps_threshold:
                    with torch.no_grad():
                        # t.max(1) will return the largest column value of each row.
                        # second column on max result is index of where max element was
                        # found, so we pick action with the larger expected reward.
                        return policy_net(state).max(1).indices.view(1, 1)
                else:
                    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            env = None
            policy_net = None

        # time stepping setup
        nt = int(np.ceil(tf / dt))
        times = np.linspace(0.0, nt * dt, nt + 1)
        snapshots = []
        c_is_callable = callable(c)

        def baseline_fgmres(A_matrix_csc, rhs_vec):
            gmres_baseline = FlexibleGMRES_original(A_matrix_csc, max_iter=config["max_iter"], tol=config["target_tol"])
            x_baseline, resid_baseline = gmres_baseline.solve(rhs_vec)
            return x_baseline, resid_baseline

        # local helper: run RL-controlled FGMRES for one linear system (A x = b)
        def rl_solve(A_matrix_csc, rhs_vec):
            try:
                solver = FlexibleGMRES_RL(A_matrix_csc, max_iter=rl_max_iter, tol=rl_tol)
                solver.initialize(rhs_vec)
                state, info = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                x_sol = None
                for it in range(rl_max_iter):
                    # choose action using epsilon-greedy
                    action = select_action(state)
                    # environment step
                    observation, x_approx, omega_list, reward, done, residual_list, time_list = env.step(action.item(), A_matrix_csc, solver)
                    reward = torch.tensor([reward], device=device)
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    # store transition
                    memory.push(state, action, next_state, reward)
                    # move to next state
                    state = next_state
                    # optimize policy network
                    optimize_model(Transition, memory, policy_net, target_net, optimizer, device, BATCH_SIZE, GAMMA,)
                    # soft update target network
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = (
                            policy_net_state_dict[key] * TAU
                            + target_net_state_dict[key] * (1 - TAU)
                        )
                    target_net.load_state_dict(target_net_state_dict)
                    x_sol = x_approx
                    if done:
                        break
                return x_sol, residual_list
            except Exception as e:
                print("RL solver failed, falling back to LU:", e)
                return None

        # ---------- time stepping ----------
        if not c_is_callable:
            c0 = float(c)
            L_const = build_L(c0)
            left_const = (I - 0.5 * dt * L_const).tocsc()
            right_const = (I + 0.5 * dt * L_const).tocsc()

            # Pre-factor LU for fallback (and to use if RL is disabled)
            LU_const = None
            if not use_rl:
                LU_const = splu(left_const)

            for k, t in enumerate(times[:-1]):
                rhs = right_const.dot(u)

                if use_rl:
                    # RL attempt
                    rhs_vec = np.asarray(rhs).ravel()
                    x_baseline, resid_baseline = baseline_fgmres(left_const, rhs_vec)
                    u_new, resid_rl = rl_solve(left_const, rhs_vec)
                    plot_fgmres_comparison(resid_baseline, resid_rl)
                    if u_new is None:
                        # fallback to LU
                        if LU_const is None:
                            LU_const = splu(left_const)
                        u = LU_const.solve(rhs)
                    else:
                        u = u_new
                else:
                    # original direct solve
                    u = LU_const.solve(rhs)

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

                rhs = right.dot(u)

                if use_rl:
                    rhs_vec = np.asarray(rhs).ravel()
                    x_baseline, resid_baseline = baseline_fgmres(left, rhs_vec)
                    u_new, resid_rl = rl_solve(left, rhs_vec)
                    plot_fgmres_comparison(resid_baseline, resid_rl)
                    if u_new is None:
                        # fallback: LU solve
                        LU = splu(left)
                        u = LU.solve(rhs)
                    else:
                        u = u_new
                else:
                    LU = splu(left)
                    u = LU.solve(rhs)

                if k % save_every == 0:
                    snapshots.append(u.copy())

        snapshots.append(u.copy())
        return {'times': times, 'solutions': snapshots, 'u_final': u}

    # Run tests / experiments
    if config.get('pure_advection_test', 0) == 1:
        print('Running pure advection problem ......')
        result = trapezoidal_method(mu=0.0, dt=dt, tf=tf, c=c, u=u, save_every=save_every)
        u_ref = np.sin(np.pi * (X - (config.get('c',1.0)) * tf)) * np.sin(np.pi * Y)
        u_ref = u_ref.ravel()
        u = result['u_final']
        solution_accuracy_test(u_ref, u, n)

    elif config.get('pure_diffusion_test', 0) == 1:
        print('Running pure diffusion problem ......')
        result = trapezoidal_method(mu=mu, dt=dt, tf=tf, c=0.0, u=u, save_every=save_every)
        # use discrete eigenvalue for reference if desired (not shown here)
        factor = np.exp(-2.0 * np.pi**2 * mu * result['times'][-1])
        u_ref = factor * u0_analytic
        u = result['u_final']
        solution_accuracy_test(u_ref, u, n)

    else:
        print('Running advection diffusion problem ......')
        result = trapezoidal_method(mu=mu, dt=dt, tf=tf, c=c, u=u, save_every=save_every)
        u = result['u_final']

    # Plot final solution:
    m = int(np.round(np.sqrt(n)))
    if m * m == n:
        U = u.reshape((m, m))
        plt.figure()
        plt.imshow(U, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(label='u')
        plt.title(f'Final solution u(T) — 2D ({m}x{m})')
    else:
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


def plot_fgmres_comparison(res_baseline, res_rl):
    plt.figure()
    plt.semilogy(res_baseline, label="FGMRES (no preconditioner)", linewidth=2)
    plt.semilogy(res_rl, label="RL-FGMRES", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Residual norm")
    plt.title("FGMRES Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("fgmres_comparison.png", dpi=300)
    plt.show()