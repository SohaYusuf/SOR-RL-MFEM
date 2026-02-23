import gym
import numpy as np
import time
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator, spilu, spsolve, lsqr
from scipy.stats import linregress
import random
from sklearn.linear_model import LinearRegression
from gym.utils import seeding 

from functions.preconditioners import M_sor
from functions.fgmres import FlexibleGMRES_RL, FlexibleGMRES_original

# from pyamg.krylov import fgmres

import pdb

from functions.utils import matrix_to_graph

class FMGRESEnv(gym.Env):
    
    def __init__(self, config={}, seed=42):
        super(FMGRESEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(config["n_actions"]) 

        # low = np.array([0.0, 0.0], dtype=np.float32)
        # high = np.array([np.inf, 1.0], dtype=np.float32) 
        # self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)  

        low  = np.array([-np.inf, -np.inf, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([ np.inf,  np.inf, np.inf, 1.0, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32) 

        self.train_loader = config["train_loader"]
        self.target_tol = config["target_tol"]
        self.omega_min = config["omega_min"]
        self.omega_max = config["omega_max"]
        self.max_iter = config["max_iter"]

    def reset(self, seed=42):
        # Ensure to pass the seed to the superclass
        super().reset(seed=seed) 
        # self.A, _, _, self.b = next(iter(self.train_loader))
        # self.solver = FlexibleGMRES_RL(self.A, max_iter=self.max_iter, tol=self.target_tol)
        # self.state =  matrix_to_graph(self.A)

        self.state = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.M = None
        self.residuals_list = []
        # self.omega = 1.0
        self.omega_list = []
        return np.array(self.state, dtype=np.float32), {}  

    def step_convergence_ratio(self):
        if len(self.residuals_list) == 1:
            return 1e-10  # No convergence rate can be computed with only one residual
        r_1 = self.residuals_list[-1]
        r_2 = self.residuals_list[-2]
        print('r_1: ',r_1)
        print('r_2: ',r_2)
        r = np.absolute(r_1 / r_2)  # Relative residual
        convergence_rate = -np.log(r)
        return convergence_rate

    def save_info(self, filename, save_path):
        """
        Saves key environment information to a text file.
        """
        # Get number of actions if using a Discrete action space
        if hasattr(self.action_space, "n"):
            num_actions = self.action_space.n
        else:
            num_actions = "Non-discrete action space"
        
        # Call reset to obtain reset values (this will update the environment's state)
        reset_state, _ = self.reset()
        
        # Build the information string
        info = (
            f"Number of Actions: {num_actions}\n"
            f"Number of observations: {len(self.state)}\n"
            f"States: {'np.log(convergence_rate), np.log(residual_norm)'}\n"
            f"Reset State: {reset_state}\n"
        )
        
        # Write the information to the specified text file
        with open(f'{save_path}/{filename}', "w") as file:
            file.write(info)
        print(f"Environment information saved to {filename}")

    
    def step(self, action, A, solver):

        print('======================> action <=======================: ', action)
        N = A.shape[0]

        parameters = list(np.linspace(self.omega_min, 
                                      self.omega_max, 
                                      num=self.action_space.n + 2)[1:-1])
        omega = parameters[action]
        self.omega_list.append(omega)

        M_sor_ = M_sor(A, omega=omega)
        _, residual_vector, _ , residual_norm, _ = solver.step(M=M_sor_, omega=omega)
    
        self.residuals_list.append(residual_norm)
        convergence_rate = self.step_convergence_ratio()

        self.state = (convergence_rate, 
                      -np.log(residual_norm), 
                      A.nnz, 
                      A.nnz / (N*N), 
                      np.mean(np.abs(A.diagonal())))
        
        print(f'\n state: {self.state}\n')
        self.state_info = str((convergence_rate, np.log(residual_norm)))

        reward = convergence_rate

        if residual_norm <= self.target_tol:
            reward -= 0.1*len(self.residuals_list)

        terminated = bool(
                residual_norm <= self.target_tol)
        
        print(f'@@ action: {action}, relaxation parameter: {omega}')
        print('@@ current_residual_norm:', residual_norm)
        print('@@ convergence_rate: ', convergence_rate)
        print(f'@@ reward: {reward}')

        return np.array(self.state, dtype=np.float32), \
            self.omega_list, reward, terminated, self.residuals_list, None
    
