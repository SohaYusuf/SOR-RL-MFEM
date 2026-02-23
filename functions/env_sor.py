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

from functions.sor import sor

# from pyamg.krylov import fgmres

import pdb

class SorEnv(gym.Env):
    
    def __init__(self, n_actions, target_tol, seed=None, all_residuals_dict=None):
        super(SorEnv, self).__init__()

        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([np.inf, 1.0], dtype=np.float32)
       
        self.action_space = gym.spaces.Discrete(n_actions)  
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)             
        self.all_residuals_dict = all_residuals_dict
        self.target_tol = target_tol

    def reset(self, seed=None):
        # Ensure to pass the seed to the superclass
        super().reset(seed=seed) 
        self.state = np.array([1.0, 1.0], dtype=np.float32)
        self.residuals_list = []
        self.current_parameter = 1e-6
        self.omega_list = []
        return np.array(self.state, dtype=np.float32), {}  

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


    def asymptotic_convergence(self, seq, tail=5, eps=1e-30):
        """Estimate asymptotic rate from seq = [||r0||,||r1||,...].
        Uses a log-linear fit (slope) on the last `tail` values only.
        Returns {'rho': exp(slope), 'slope': slope, 'rho_tail': last_ratios}."""
        s = np.asarray(seq, dtype=float).ravel()
        if s.size < 2:
            raise ValueError("seq must have at least 2 entries")
        s = np.maximum(s, eps)
        # last `tail` points (at least 2 needed)
        t = min(int(tail or 5), s.size)
        if t < 2:
            raise ValueError("tail must be >= 2")
        k = np.arange(s.size - t, s.size)
        y = np.log(s[-t:])
        # linear fit y = intercept + slope * k
        A = np.vstack([np.ones_like(k), k]).T
        intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]
        ratios = s[1:] / s[:-1]
        tail_vals = ratios[-max(1, t-1):]  # last t-1 ratios correspond to last t points
        return {'rho': float(np.exp(slope)), 'slope': float(slope), 'rho_tail': tail_vals}


    
    def step(self, action, A, b):

        print('==============> action <==============: ', action)
        N = A.shape[0]
        
        parameters = list(np.linspace(0.0, 2.0, num=self.action_space.n + 2)[1:-1])
        omega = parameters[action]
    
        print('======================> omega [0,2] <=======================: ', omega)
        residual_list = self.all_residuals_dict['residuals_SOR'][N][omega]
        num_iter = len(residual_list)
        convergence_rate = self.asymptotic_convergence(residual_list)['rho']

        self.state = (convergence_rate, num_iter) 
        self.state_info = str((convergence_rate, num_iter))
        reward = -num_iter

        print(f'\n state: {self.state}\n')
        print(f'number of iterations: {num_iter}')
        print(f'\nreward: {reward}\n')

        # reward = convergence_rate
        terminated = bool(
                residual_list[-1] <= self.target_tol)
       
        return np.array(self.state, dtype=np.float32), [omega], \
            reward, terminated, residual_list, None
