# python main.py --num_episodes 100 --data_size_list 256 576 144 1024 2304 --default_omega 0.2 --run_FGMRES_baseline 0 --run_FGMRES_default_SOR 0 --train_RL_model 0

# Standard library

import json
import os
import pdb

# Third-party
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from scipy.sparse import csc_matrix

from functions.env_fgmres import FMGRESEnv
from functions.env_sor import SorEnv
from functions.fgmres import FlexibleGMRES_RL
from functions.read_data_advection import get_test_loader
from functions.run_baselines_fgmres_advection import run_baselines_fgmres_and_plot
from functions.sor import sor
from functions.utils import compute_omega_opt, estimate_asymptotic_cr, plot_results_dynamic

def _greedy_action(state, net, device):
    """Return scalar action (torch.int) chosen greedily by policy net."""
    with torch.no_grad():
        out = net(state.to(device))
        # handle nets that return tuples (e.g., (q_vals, other))
        if isinstance(out, tuple): out = out[0]
        if out.dim() == 1: out = out.unsqueeze(0)
        action = out.argmax(dim=1).cpu().item()
    return action

def test_policy_fgmres(policy_net,
                config,
                device='cpu',
                num_episodes=5,
                seed=42):
    """
    Evaluate a trained policy_net deterministically.
    - policy_net: PyTorch model instance (same architecture used in training).
    - weights_path: path to saved state_dict (.pth).
    - data_loader: yields (A, A_tensor, x_true, b) per batch/sample.
    - config: dictionary containing at least 'n_actions' and 'target_tol'.
    - save_path: directory to save outputs (residuals, omegas).
    - device: 'cpu' or torch device.
    - num_episodes: episodes to run per sample in data_loader.
    Returns (residuals_RL, omegas_over_episodes).
    """
    data_loader = get_test_loader(test_path=config["test_data_path"], 
                                    mode=config["mode"], 
                                    device=device)
    
    save_path = config.get("save_path", "results")
    save_path = f'{save_path}/FGMRES/'
    os.makedirs(save_path, exist_ok=True)
    test_results_folder = f'{save_path}/test_results/'
    os.makedirs(test_results_folder, exist_ok=True)


    fname = os.path.join(test_results_folder, "all_residuals_dict_baselines_fgmres.npz")

    if os.path.exists(fname):
        npz = np.load(fname, allow_pickle=True)
        loaded = {}
        for k in npz.files:
            v = npz[k]
            loaded[k] = v.item() if v.shape == () else v
        all_residuals_dict = loaded["all_residuals"]
        # restore other items if you need them locally:
        optimal_omegas = loaded.get("optimal_omegas", None)
        plotted_lines = loaded.get("plotted_lines", None)
        metrics_df = loaded.get("metrics_df", None)
        print(f"Loaded all_residuals_dict (and wrapper) from {fname}")
    else:
        all_residuals_dict = run_baselines_fgmres_and_plot(config, data_loader, test_results_folder)

        wrapper = {
            "residuals_baseline": all_residuals_dict.get("baseline"),
            "residuals_SOR": all_residuals_dict.get("SOR"),
            "plotted_lines": all_residuals_dict.get("plotted_lines"),
            "all_residuals": all_residuals_dict,
            "optimal_omegas": locals().get("optimal_omegas", None),
            "metrics_df": locals().get("df_metrics", None)
        }

        # save everything in one .npz (pickles non-array objects)
        np.savez_compressed(fname, **wrapper)
        print(f"Saved all_residuals_dict (and wrapper) to {fname}")

    last_episode = config["num_episodes"]-1
    weights = f'{save_path}/policy_checkpoints/policy_net_weights_{last_episode}.pth'  # or final checkpoint

    # load weights and set eval mode
    policy_net.load_state_dict(torch.load(weights, map_location=device))
    policy_net.to(device)
    policy_net.eval()

    env = FMGRESEnv(config=config)

    residuals_RL = {}
    omegas_over_episodes = []
    episode_counter = 0

    for (A, A_tensor, x_true, b) in data_loader:
        if config["dataset"]=="diffusion":
            omega_opt_symmetric = compute_omega_opt(A)
        else:
            omega_opt_symmetric = None
        A = csc_matrix(A)            # match training conversion
        N = A.shape[0]
        solver = FlexibleGMRES_RL(A, max_iter=N, tol=config["target_tol"])

        for ep in range(num_episodes):
            episode_counter += 1
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            solver.initialize(b)

            # run deterministic episode (greedy policy)
            while True:
                action = _greedy_action(state, policy_net, device)
                observation, omega_list, reward, done, residual_list, time_list = env.step(action, A, solver)
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                if done:
                    residuals_RL[N] = residual_list
                    # store residual history (exclude last if desired; match your training code)
                    # residuals_RL.setdefault(N, []).append([float(r) for r in residual_list])
                    all_residuals_dict['residuals_RL'] = residuals_RL
                    # print short summary
                    final_res = residual_list[-1] if len(residual_list) else None
                    print(f"[Test] sample N={N}, ep={ep}, last_omega={omega_list[-1]:.6e}, final_res={final_res:.3e}")

                    plot_results_dynamic(all_residuals_dict, 
                                 N, 
                                 rl_omega=omega_list, 
                                 opt_omega=omega_opt_symmetric, 
                                 save_path=f'{test_results_folder}/test_omega_dynamic_N{N}.png',
                                 target_tol=config["target_tol"])

                    break
            

    # save results
    np.save(os.path.join(test_results_folder, "omegas_over_episodes_test.npy"), np.array(omegas_over_episodes))
    with open(os.path.join(test_results_folder, "residuals_RL_test.json"), "w") as f:
        json.dump(residuals_RL, f, indent=2)

    print(f"Testing done: {len(omegas_over_episodes)} episodes evaluated. Results saved to {test_results_folder}")
    return residuals_RL, omegas_over_episodes


def test_policy_sor(policy_net,
                data_loader,
                all_residuals_dict,
                config,
                device='cpu',
                num_episodes=5,
                seed=42):
    """
    Evaluate a trained policy_net deterministically.
    - policy_net: PyTorch model instance (same architecture used in training).
    - weights_path: path to saved state_dict (.pth).
    - data_loader: yields (A, A_tensor, x_true, b) per batch/sample.
    - config: dictionary containing at least 'n_actions' and 'target_tol'.
    - save_path: directory to save outputs (residuals, omegas).
    - device: 'cpu' or torch device.
    - num_episodes: episodes to run per sample in data_loader.
    Returns (residuals_RL, omegas_over_episodes).
    """

    pdb.set_trace()
    save_path = config.get("save_path", "results")
    save_path = f'{save_path}/SOR/'
    os.makedirs(save_path, exist_ok=True)
    test_results_folder = f'{save_path}/test_results/'
    os.makedirs(test_results_folder, exist_ok=True)

    last_episode = config["num_episodes"]-1
    weights = f'{save_path}/policy_checkpoints/policy_net_weights_{last_episode}.pth'  # or final checkpoint

    # load weights and set eval mode
    policy_net.load_state_dict(torch.load(weights, map_location=device))
    policy_net.to(device)
    policy_net.eval()

    env = SorEnv(config['n_actions'], 
                 target_tol=config['target_tol'], 
                 seed=seed,
                 all_residuals_dict=all_residuals_dict)

    residuals_RL = {}
    omegas_over_episodes = []
    episode_counter = 0

    for (A, A_tensor, x_true, b) in data_loader:
        A = csc_matrix(A)            # match training conversion
        N = A.shape[0]
        # solver = FlexibleGMRES_RL(A, max_iter=N, tol=config['target_tol'])

        for ep in range(num_episodes):
            episode_counter += 1
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # solver.initialize(b)

            # run deterministic episode (greedy policy)
            while True:
                action = _greedy_action(state, policy_net, device)
                observation, omega_list, reward, done, residual_list, time_list = env.step(action, A, b)

                # pdb.set_trace()
                test_omega = omega_list[-1]
                print(f'\n Test omega: {test_omega}')
                # next state
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                if done:
                    # mean_omega = float(np.mean(omega_list)) if len(omega_list) > 0 else float('nan')
                    omegas_over_episodes.append(test_omega)
                    # omegas_over_episodes.append(omega_list[-1])

                    # store residual history (exclude last if desired; match your training code)
                    residuals_RL.setdefault(N, []).append([float(r) for r in residual_list])

                    # print short summary
                    final_res = residual_list[-1] if len(residual_list) else None
                    print(f"[Test] sample N={N}, ep={ep}, last_omega={omega_list[-1]:.6e}, final_res={final_res:.3e}")

                    break

    # save results
    np.save(os.path.join(save_path, "omegas_over_episodes.npy"), np.array(omegas_over_episodes))
    with open(os.path.join(save_path, "residuals_RL.json"), "w") as f:
        json.dump(residuals_RL, f, indent=2)

    print(f"Testing done: {len(omegas_over_episodes)} episodes evaluated. Results saved to {save_path}")

    test_omega = np.mean(omegas_over_episodes)
    print(f'\n @@ Test mean omega: {test_omega} \n')
    
    num_iters_test, x_SOR_test, residuals_RL_test = sor(A, 
                                                        b, 
                                                        omega=test_omega, 
                                                        tol=config["target_tol"], 
                                                        max_iter=250)

    # Calculate solution error with SOR
    solution_error_SOR_test = np.sqrt(mean_squared_error(x_true, x_SOR_test))
    print(f'solution_error with SOR: {solution_error_SOR_test}')
    print(f'residuals using test SOR: {residuals_RL_test}')
    N_test = A.shape[0]
    
    optimal_omega = config["default_omega"]
    optimal_omega_residuals = all_residuals_dict["residuals_SOR"][N][optimal_omega]

    rl_cr = estimate_asymptotic_cr(residuals_RL_test)
    opt_cr = estimate_asymptotic_cr(optimal_omega_residuals)

    print(f'\n @@ Asymptotic convergence rate for RL test_omega {test_omega}: {rl_cr}\n')
    print(f'\n @@ Asymptotic convergence rate for optimal omega {optimal_omega}: {opt_cr}\n')
    

    save_path_figure=f'{test_results_folder}/test_omega_{test_omega:.4f}.png'
    fontsize = 20
    plt.figure(figsize=(6, 4))
    ax = plt.gca(); ax.set_facecolor('#F5F5F5')
    ax.semilogy(residuals_RL_test, '--', lw=2, label=f'ω={test_omega:.4f} (ours)', color='blue')
    ax.semilogy(optimal_omega_residuals, '-.', lw=2, label=f'ω={optimal_omega:.4f} (theoretical)', color='red')
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Residual norm', fontsize=fontsize)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', ls='--', lw=0.5)
    plt.xticks(fontsize=fontsize); plt.yticks(fontsize=fontsize)
    plt.savefig(save_path_figure, dpi=300, bbox_inches='tight')
    plt.close()
    
    return residuals_RL, omegas_over_episodes