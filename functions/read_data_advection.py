import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

import pdb
import re


def custom_collate_fn(batch):
    """
    Since the DataLoader batch_size is always 1, this collate function simply
    returns the sole element from the batch list without further processing.
    """
    return batch[0]

def generate_rhs(A, num_rhs=5, seed=None):
    rng = np.random.default_rng(seed)
    Bs = []
    for _ in range(num_rhs):
        u = rng.standard_normal(A.shape[0])
        b = A.dot(u)
        Bs.append((u, b))
    return Bs


class LinearSystemDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, sizes, mode, device, p_list=[3]):
        self.train_path = train_path
        self.sizes = sizes
        self.device = device
        self.data = []
        self.mode = "train"
        data = read_A_file(self.train_path)

        # Diffusion system
        A_d = data['Diffusion']['A_d']
        u_d = data['Diffusion']['u_d']
        b_d = data['Diffusion']['b_d']

        # Advection system
        A_c = data['Advection']['A_c']
        u_c = data['Advection']['u_c']
        b_c = data['Advection']['b_c']

        # store both systems
        self.data.append((A_d, u_d, b_d))
        self.data.append((A_c, u_c, b_c))

        # for N in sizes:
        #     for p in p_list:  
        #         A, A_tensor, x_true, b = read_A_file(self.train_path, 
        #                                              N, 
        #                                              p, 
        #                                              self.mode, 
        #                                              self.device)
        #         self.data.append((A, A_tensor, x_true, b))
                # if mode=="train":
                #     # 2) Add multiple RHS per matrix
                #     for (u_rand, b_rand) in generate_rhs(A, num_rhs=200):
                #         self.data.append((A, A_tensor, u_rand, b_rand))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_train_loader(train_path, batch_size, mode, device):
    sizes=[144]
    dataset = LinearSystemDataset(train_path, sizes, mode, device=device, p_list=[0])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return train_loader

def get_test_loader(test_path, mode, device):
    batch_size=1
    # sizes = [144,576,2304,9216]
    sizes = [144]
    dataset = LinearSystemDataset(test_path, sizes, mode, device, p_list=[0])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader

def read_mfem_vector(path):
    vals = []
    with open(path,'r') as f:
        for line in f:
            for tok in line.strip().split():
                vals.append(float(tok))
    return np.array(vals)


def read_data(train_path):
    """
    Load advection (M_ex9_, K_ex9_, u_ex9_, b_ex9_) and diffusion
    (A_ex14_, u_ex14_, b_ex14_) files from train_path and return:
      {'Diffusion':{'A_d':A_d, 'u_d':u_d, 'b_d':b_d},
       'Advection':{'A_c':A_c, 'u_c':u_c, 'b_c':b_c}}
    Assumes modules: os, re, numpy as np, and coo_matrix are available.
    """

    float_re = r'[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?'
    param_re = re.compile(rf'N=(\d+)(?:_dt=({float_re}))?(?:_CFL=({float_re}))?')

    # containers for results
    M = K = A_d = None
    u_c = u_d = b_c = b_d = None
    N = dt = CFL = None

    # helper to strip Windows ADS and extension
    clean = lambda fn: os.path.splitext(fn.split(':', 1)[0])[0]

    for fname in sorted(os.listdir(train_path)):
        base = clean(fname); path = os.path.join(train_path, fname)

        # parse parameters if present
        m = param_re.search(base)
        if m:
            N = int(m.group(1)) if m.group(1) else N
            dt = float(m.group(2)) if m.group(2) else dt
            CFL = float(m.group(3)) if m.group(3) else CFL

        # advection matrices/vectors
        if base.startswith('M_ex9_'):
            d = np.loadtxt(path); r, c, v = d[:,0].astype(int), d[:,1].astype(int), d[:,2]
            M = coo_matrix((v, (r-1, c-1))); print('Loaded M:', fname)

        elif base.startswith('K_ex9_'):
            d = np.loadtxt(path); r, c, v = d[:,0].astype(int), d[:,1].astype(int), d[:,2]
            K = coo_matrix((v, (r-1, c-1))); print('Loaded K:', fname)

        elif base.startswith('u_ex9_'):
            # always skip the first 5 header lines for u files
            u_c = np.loadtxt(path, skiprows=5); print('Loaded u (advection):', fname)

        # diffusion files
        elif base.startswith('A_ex14'):
            d = np.loadtxt(path)
            if d.ndim == 2 and d.shape[1] >= 3:
                r, c, v = d[:,0].astype(int), d[:,1].astype(int), d[:,2]
                A_d = coo_matrix((v, (r-1, c-1)))
            else:
                A_d = coo_matrix(d)
            print('Loaded A (diffusion):', fname)

        elif base.startswith('u_ex14'):
            u_d = np.loadtxt(path, skiprows=5); print('Loaded u (diffusion):', fname)
            print(u_d)

    # validate advection inputs
    if M is None or K is None:
        raise RuntimeError("Missing M_ex9_ or K_ex9_ files in folder.")
    if dt is None:
        raise RuntimeError("Could not parse dt from filenames (required for advection A).")

    # build advection operator and return both datasets
    A_c = coo_matrix(M - (dt/2.0) * K)

    b_c = A_c.dot(u_c)
    b_d = A_d.dot(u_d)

    data = {
        'Diffusion': {'A_d': A_d, 'u_d': u_d, 'b_d': b_d},
        'Advection': {'A_c': A_c, 'u_c': u_c, 'b_c': b_c, 'M':M, 'K':K}
    }

    A_d = A_d.tocsr()
    A_c = A_c.tocsr()

    print(f'Error in ||A_d*u_d - b_d||: {np.linalg.norm(b_d - A_d.dot(u_d))}')
    print(f'Error in ||A_c*u_c - b_c||: {np.linalg.norm(b_c - A_c.dot(u_c))}')

    # helper to convert scipy coo_matrix -> torch.sparse_coo_tensor
    def coo_to_torch_sp(mat):
        if mat is None:
            return None
        coo = mat.tocoo()
        idx = np.vstack((coo.row, coo.col))
        idx_t = torch.LongTensor(idx)               # shape (2, nnz)
        vals_t = torch.tensor(coo.data, dtype=torch.float32)
        size = tuple(coo.shape)
        return torch.sparse_coo_tensor(idx_t, vals_t, size)

    # convert vectors -> torch dense tensors (1D)
    def vec_to_torch(v):
        return None if v is None else torch.tensor(np.ravel(v), dtype=torch.float32)

    # assemble tensor dict (sparse tensors for matrices)
    data_tensor = {
        'Diffusion': {'A_d': coo_to_torch_sp(A_d), 'u_d': vec_to_torch(u_d), 'b_d': vec_to_torch(b_d)},
        'Advection': {'A_c': coo_to_torch_sp(A_c), 'u_c': vec_to_torch(u_c), 'b_c': vec_to_torch(b_c)}
    }

    print(f"Parsed params: N={N}, dt={dt}, CFL={CFL}")

    return data, data_tensor







# ----------------------------------------------------------
# def read_A_file(train_path, N, p, mode, device):
#     """
#     Reads a sparse matrix from a text file and returns:
#     - A: scipy sparse coo_matrix
#     - A_tensor: torch sparse coo tensor
#     - u_mfem: vector from u file
#     - b_mfem: matrix-vector product A * u_mfem
#     """

#     # steady advection equation matrix A_c such that A_c u = 0 
#     # A_c u = 0 is solved using MFEM 
#     data = compute_A_advection(train_path)

#     pdb.set_trace()

#     # Define file paths
#     A_d_path = os.path.join(train_path, f"A_ex14_N={N}.txt")
#     u_d_path = os.path.join(train_path, f"u_ex14_N={N}.txt")

#     # Read sparse matrix entries from the text file
#     sparse_matrix_entries_ = []
#     with open(A_d_path, 'r') as file:
#         for line in file:
#             row, col, val = map(float, line.split())
#             sparse_matrix_entries_.append((int(row), int(col), val))   # changed this for ex0.cpp
#     # Extract row indices, column indices, and values
#     rows, cols, vals = zip(*sparse_matrix_entries_)

#     # Construct scipy sparse matrix
#     A_d = coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
#     print(f'A shape: \n{A.shape}\nNo. of entries: {len(vals)}')

#     ################################################ added by soha for using ex1.cpp #################################
#     #################################################################################################################

#     # Convert A to CSR format (efficient for matrix multiplication)
#     A_d = A_d.tocsr()

#     # Compute b_mfem
#     # u_mfem = np.loadtxt(u_txt_file_path)
#     u_d = read_mfem_vector(u_d_path)
#     print('u_d shape: ',u_d.shape)

#     b_d = A_d.dot(u_d)
#     b_c = A_c.dot(u_c)
#     print(f'b_d shape: {b_d.shape}')
#     print(f'b_c shape: {b_c.shape}')
#     print(f'Error in ||A_d*u_d - b_d||: {np.linalg.norm(b_d - A_d.dot(u_d))}')
#     print(f'Error in ||A_c*u_c - b_c||: {np.linalg.norm(b_c - A_c.dot(u_c))}')

#     pdb.set_trace()
#     # Construct sparse coo tensor in torch
#     vals_tensor = torch.tensor(vals, dtype=torch.float64).to(device)
#     rows_tensor = torch.tensor(rows, dtype=torch.int64).to(device)
#     cols_tensor = torch.tensor(cols, dtype=torch.int64).to(device)
#     A_d_tensor = torch.sparse_coo_tensor(indices=torch.stack([rows_tensor, cols_tensor]), 
#                                        values=vals_tensor, 
#                                        size=(N, N), 
#                                        dtype=torch.float64).to(device)
    
#     data = {'Diffusion':{'A_d':A_d, 'u_d':u_d, 'b_d':b_d}, 
#             'Advection':{'A_c':A_c, 'u_c':u_c, 'b_c':b_c}}

#     return A, A_d_tensor, u_d, b_d