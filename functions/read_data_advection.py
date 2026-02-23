import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

import pdb


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


        for N in sizes:
            for p in p_list:  
                A, A_tensor, x_true, b = read_A_file(self.train_path, 
                                                     N, 
                                                     p, 
                                                     self.mode, 
                                                     self.device)
                self.data.append((A, A_tensor, x_true, b))
                # if mode=="train":
                #     # 2) Add multiple RHS per matrix
                #     for (u_rand, b_rand) in generate_rhs(A, num_rhs=200):
                #         self.data.append((A, A_tensor, u_rand, b_rand))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_train_loader(train_path, batch_size, mode, device):
    sizes=[576]
    dataset = LinearSystemDataset(train_path, sizes, mode, device=device, p_list=[0])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return train_loader

def get_test_loader(test_path, mode, device):
    batch_size=1
    sizes = [144,576,2304,9216]
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



def read_A_file(train_path, N, p, mode, device):
    """
    Reads a sparse matrix from a text file and returns:
    - A: scipy sparse coo_matrix
    - A_tensor: torch sparse coo tensor
    - u_mfem: vector from u file
    - b_mfem: matrix-vector product A * u_mfem
    """

    # Define file paths
    txt_file_path = os.path.join(train_path, f"A_{N}_p{p}.txt")
    u_txt_file_path = os.path.join(train_path, f"u_{N}_p{p}.txt")

    # Read sparse matrix entries from the text file
    sparse_matrix_entries_ = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            row, col, val = map(float, line.split())
            sparse_matrix_entries_.append((int(row), int(col), val))   # changed this for ex0.cpp
    # Extract row indices, column indices, and values
    rows, cols, vals = zip(*sparse_matrix_entries_)

    # Construct scipy sparse matrix
    A = coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    print(f'A shape: \n{A.shape}\nNo. of entries: {len(vals)}')

    ################################################ added by soha for using ex1.cpp #################################
    #################################################################################################################

    # Convert A to CSR format (efficient for matrix multiplication)
    A = A.tocsr()

    # Compute b_mfem
    # u_mfem = np.loadtxt(u_txt_file_path)
    u_mfem = read_mfem_vector(u_txt_file_path)
    print('u_mfem shape: ',u_mfem.shape)

    b_mfem = A.dot(u_mfem)
    print(f'b_mfem shape: {b_mfem.shape}')
    print(f'Error in ||A*u_mfem - b_mfem||: {np.linalg.norm(b_mfem - A.dot(u_mfem))}')

    # Construct sparse coo tensor in torch
    vals_tensor = torch.tensor(vals, dtype=torch.float64).to(device)
    rows_tensor = torch.tensor(rows, dtype=torch.int64).to(device)
    cols_tensor = torch.tensor(cols, dtype=torch.int64).to(device)
    A_tensor = torch.sparse_coo_tensor(indices=torch.stack([rows_tensor, cols_tensor]), values=vals_tensor, size=(N, N), dtype=torch.float64).to(device)

    return A, A_tensor, u_mfem, b_mfem