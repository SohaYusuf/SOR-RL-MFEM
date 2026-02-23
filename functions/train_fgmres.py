# python main.py --run_sor_as_solver 1 

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from sklearn.metrics import mean_squared_error
import json
import os
import numpy as np
from scipy.sparse import csc_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

from functions.env_fgmres import FMGRESEnv
from functions.fgmres import FlexibleGMRES_RL
from functions.model import DQN, optimize_model
from functions.plot_policy import plot_policy_heatmap
from functions.read_data_advection import get_test_loader
from functions.utils import device, plot_durations, plot_omega_over_episodes, plot_results_dynamic, plot_rewards


def train_fgmres(config):

    # Enable interactive plotting mode and configure for IPython
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    print('\n Running main ..... \n')

    # Extract relevant command line arguments from the configuration
    args = {k: config[k] for k in ["seed", "batch_size", "gamma", "eps_start", "eps_end", "eps_decay",
                               "tau", "learning_rate", "num_episodes", "target_tol", "default_omega",
                               "omega_min", "omega_max", "max_iter", 
                               "debug", "train_data_path", "test_data_path", "save_path", "dataset", "checkpoint", "mode",
                               "n_actions", "train_RL_model", "run_FGMRES_baseline", "run_FGMRES_default_SOR",
                               "run_sor_as_solver", 
                               "run_fgmres_as_solver"]}

    print(args)
    
    if config["dataset"]=='advection':
        from functions.read_data_advection import get_train_loader
        from functions.run_baselines_fgmres_advection import run_baselines_fgmres_and_plot
    if config["dataset"]=='diffusion':
        from functions.read_data_diffusion import get_train_loader
        from functions.run_baselines_fgmres import plot_baseline_residuals_fgmres, run_baselines_fgmres, run_baselines_fgmres_and_plot
        
    # Set up the directory to save results
    save_path = config.get("save_path", "results")
    save_path = f'{save_path}/FGMRES/'
    os.makedirs(save_path, exist_ok=True)
    seed = config["seed"]
    ddtype = np.float32

    # Convert the dictionary into a list of key-value pairs, converting values to strings
    config_list = [[key, str(value)] for key, value in args.items()]

    # Convert the list to a NumPy array
    config_array = np.array(config_list)

    # Save the configuration array to a text file using np.savetxt
    # fmt='%s' ensures all entries are treated as strings,
    # and the delimiter separates the key and value with ": "
    np.savetxt(f"{save_path}/config.txt", config_array, fmt="%s", delimiter=": ")

   
    train_loader = get_train_loader(train_path=config["train_data_path"], 
                                    batch_size=config["batch_size"], 
                                    mode=config["mode"], 
                                    device=device)
    
    test_loader = get_train_loader(train_path=config["train_data_path"], 
                                    batch_size=config["batch_size"], 
                                    mode="test", 
                                    device=device)
    
    config["train_loader"] = train_loader
    
    # run the solver without reinforcement learning framework
    
    fname = os.path.join(save_path, "all_residuals_dict_baselines_fgmres.npz")

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
        all_residuals_dict = run_baselines_fgmres_and_plot(config, test_loader, save_path)

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


    # pdb.set_trace()
    # Reinforcement learning framework setup 
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
        
        
        
    if config["train_RL_model"]==True:

        episode_durations = []
        rewards_list = []
        counter_ = 0
        episode_rewards = []
        training_states_list = []
        actions_list = []
        omegas_over_episodes = []

        checkpoint_folder = f'{save_path}/policy_checkpoints/'
        os.makedirs(checkpoint_folder, exist_ok=True)

        train_results_folder = f'{save_path}/train_results/'
        os.makedirs(train_results_folder, exist_ok=True)

        torch.autograd.set_detect_anomaly(True)

        residuals_RL = {}

        env = FMGRESEnv(config=config)
        
        env.save_info(filename='env_fgmres_info.txt', save_path=save_path)

        total_episodes = num_episodes*len(train_loader)
        total_episode_counter = 0

        omegas_over_episodes = []

        # prepare training examples list (so we can sample randomly)
        train_examples = []
        for A, A_tensor, x_true, b in train_loader:
            train_examples.append((csc_matrix(A), A_tensor, x_true, b))
        

        for i_episode in range(total_episodes):


            idx = random.randrange(len(train_examples))
            A, A_tensor, x_true, b = train_examples[idx]
            N = A.shape[0]
            config["default_omega"]=all_residuals_dict["optimal_omegas"][N]

            solver = FlexibleGMRES_RL(A, max_iter=config["max_iter"], tol=config["target_tol"])

            print(f'--- Global episode {i_episode} (matrix idx {idx}, N={N}) ---')
            
            total_rewards = 0
            episode_reward = 0

            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # Initialize Flexible GMRES
            solver.initialize(b)

            if i_episode == 0 or i_episode == num_episodes // \
            3 or i_episode == (2 * num_episodes) // 3 or i_episode == num_episodes - 1:
                torch.save(policy_net.state_dict(), f'{checkpoint_folder}/policy_net_weights_{i_episode}.pth')

            for t in count():

                counter_ = counter_ + 1
                action = select_action(state)

                observation, omega_list, reward, done, residual_list, time_list = env.step(action.item(), A, solver)

                reward = torch.tensor([reward], device=device)
                
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                
                # optimize_model()
                optimize_model(Transition, memory, policy_net, target_net, optimizer, device, BATCH_SIZE, GAMMA)

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                # Save the policy_net weights
                torch.save(policy_net.state_dict(), 'policy_net_weights.pth')
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (policy_net_state_dict[key] * TAU +
                                                target_net_state_dict[key] * (1 - TAU))
                
                target_net.load_state_dict(target_net_state_dict)
                total_rewards = total_rewards + reward.item()

                # actions_list.append(action.item())
                rewards_list.append(reward.item())

                episode_reward+=reward.item()
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_durations.append(t + 1)
                    # plot_durations(episode_durations, train_results_folder, is_ipython)
                    plot_rewards(episode_rewards, train_results_folder, is_ipython, False, 20, 'training_rewards_fgmres.png')
                    break

            omegas_over_episodes.append(float(np.mean(omega_list)))
            training_states_list.append(state)
            actions_list.append(action.item())

            # residuals_RL[N] = residual_list[:-1]
            residuals_RL[N] = residual_list
            # pdb.set_trace()
            all_residuals_dict['residuals_RL'] = residuals_RL

            if i_episode % 50 == 0:
                np.savetxt(f'{train_results_folder}/residuals_{i_episode}.txt', residual_list)
                print('residual_list',residual_list)
                file_name = f'{train_results_folder}/residual_plot_{i_episode}_N{N}.png'
                # pdb.set_trace()
                plot_results_dynamic(all_residuals_dict, 
                                N, 
                                rl_omega=omega_list, 
                                opt_omega=None, 
                                save_path=file_name,
                                target_tol=config["target_tol"])

            
            total_episode_counter +=1
            print(f'episode {i_episode}, \
                    action {action.item()}  ======> state: {observation} ======> reward: {episode_reward}')

        plot_omega_over_episodes(omegas_over_episodes, 'episode', 'omega', 'omega', 
                            f'{save_path}/omega_over_episodes_fgmres_N{N}.png', config["default_omega"])
        
        plot_omega_over_episodes(omegas_over_episodes, 'episode', 'omega', 'omega', 
                            f'{save_path}/omega_over_episodes_fgmres_N{N}.png', config["default_omega"], log=True)

        print('Complete')
        plot_durations(episode_durations,train_results_folder, is_ipython, show_result=False)
        plt.ioff()
            # plt.show()

        with open(f"{save_path}/residuals_RL.txt", "w") as f:
            json.dump(residuals_RL, f, indent=4)

            
    return policy_net, all_residuals_dict