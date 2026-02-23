import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from functions.utils import classify_sparse_matrix
import pdb
from scipy.sparse.csgraph import reverse_cuthill_mckee



def custom_collate_fn(batch):
    """
    Since the DataLoader batch_size is always 1, this collate function simply
    returns the sole element from the batch list without further processing.
    """
    return batch[0]

class LinearSystemDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, mode, device):
        self.train_path = train_path
        self.device = device
        self.data = []
        self.mode = "train"

        # read all examples (read_A_file returns a list of examples)
        for p in [1]:
            examples = read_A_file(self.train_path, p, self.mode, self.device)
            # add each (A, A_tensor, u, b) tuple to dataset
            for ex in examples:
                self.data.append(ex)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_train_loader(train_path, batch_size, mode, device):
    dataset = LinearSystemDataset(train_path, mode, device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return train_loader


def read_mfem_vector(path):
    """
    Read vector from MFEM-style text file, skipping the first 5 lines (header).
    After the header, blank lines and lines starting with '%' are ignored.
    """
    vals = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            # skip the first 5 lines (i = 0..4)
            if i < 5:
                continue
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    # ignore non-numeric tokens
                    continue
    return np.array(vals, dtype=float)

def reorder_with_rcm(A, u, symmetric_mode=True):

    
    perm = reverse_cuthill_mckee(A, symmetric_mode=symmetric_mode)
    # -- Permute matrix and unknown vector --
    A_rcm = A[perm][:, perm]         # permuted sparse matrix
    u_arr = np.asarray(u).ravel()    # make 1D vector
    u_rcm = u_arr[perm]              # permuted unknowns
    b_rcm = A_rcm.dot(u_rcm)
    
    # --- plot both sparsity patterns side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    # left: original
    ax = axes[0]
    ax.spy(A, markersize=0.6)
    ax.set_title('Original A (sparsity)')
    ax.set_xlabel('column index')
    ax.set_ylabel('row index')

    # right: reordered
    ax = axes[1]
    ax.spy(A_rcm, markersize=0.6)
    ax.set_title('RCM reordered A (sparsity)')
    ax.set_xlabel('column index')
    ax.set_ylabel('row index')

    # plt.suptitle('Sparsity pattern: before and after Reverse Cuthill–McKee', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save single figure with both panels
    out_fname = 'results/spy_A_before_after_rcm.png'
    plt.savefig(out_fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_fname}")

    # pdb.set_trace()
    
    return A_rcm, u_rcm, b_rcm



def read_A_file(train_path, p, mode, device):
    """
    Reads ALL sparse matrices in train_path starting with 'diffusion_A_' and their
    corresponding 'diffusion_u_' files. Returns a list of tuples:
        [(A, A_tensor, u_mfem, b_mfem), ...]
    """

    # list files in folder
    all_files = sorted(os.listdir(train_path))

    # find A/u candidates
    A_candidates = [f for f in all_files if f.startswith("diffusion_A_")]
    u_candidates = [f for f in all_files if f.startswith("diffusion_u_")]

    if not A_candidates:
        raise FileNotFoundError(f"No files starting with 'diffusion_A_' found in {train_path}")

    results = []
    u_set = set(u_candidates)

    for a_fname in A_candidates:
        # attempt to find matching u filename by replacing prefix
        suffix = a_fname[len("diffusion_A_"):]  # everything after 'diffusion_A_'
        candidate_u = "diffusion_u_" + suffix
        if candidate_u in u_set:
            u_fname = candidate_u
        else:
            # fallback: try to match by N substring if possible, else use first u_candidate
            matched = None
            if "_N" in a_fname:
                idx = a_fname.find("_N")
                part = a_fname[idx:]  # e.g. "_N256_..." or "_N256."
                for uc in u_candidates:
                    if part in uc:
                        matched = uc
                        break
            if matched:
                u_fname = matched
            elif u_candidates:
                u_fname = u_candidates[0]
            else:
                raise FileNotFoundError(f"No corresponding 'diffusion_u_' file found for {a_fname} in {train_path}")

        txt_file_path = os.path.join(train_path, a_fname)
        u_txt_file_path = os.path.join(train_path, u_fname)

        print(f'\n A file path: {txt_file_path}')
        print(f' u file path: {u_txt_file_path} \n')

        # Read sparse matrix entries, skipping lines that start with '%' or are blank
        sparse_matrix_entries_ = []
        with open(txt_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    row_f = float(parts[0]); col_f = float(parts[1]); val_f = float(parts[2])
                except:
                    # skip malformed lines
                    continue
                sparse_matrix_entries_.append((row_f, col_f, val_f))

        if len(sparse_matrix_entries_) == 0:
            print(f"Warning: no numeric entries found in {txt_file_path} — skipping")
            continue

        # determine if indexing is 1-based or 0-based
        rows_raw = [entry[0] for entry in sparse_matrix_entries_]
        cols_raw = [entry[1] for entry in sparse_matrix_entries_]
        vals_raw = [entry[2] for entry in sparse_matrix_entries_]

        min_index = min(min(rows_raw), min(cols_raw))
        if min_index >= 1:
            # 1-based indexing (usual MFEM), convert to 0-based
            rows = [int(r - 1) for r in rows_raw]
            cols = [int(c - 1) for c in cols_raw]
        else:
            rows = [int(r) for r in rows_raw]
            cols = [int(c) for c in cols_raw]

        # infer matrix size
        N_inferred = max(max(rows), max(cols)) + 1

        # Construct scipy sparse matrix
        A = coo_matrix((vals_raw, (rows, cols)), shape=(N_inferred, N_inferred), dtype=np.float64)
        print(f'A shape: \n{A.shape}\nNo. of entries: {len(vals_raw)}')

        # Convert A to CSR format (efficient for matrix multiplication)
        A = A.tocsr()

        print(f'\n')
        res = classify_sparse_matrix(A)
        print(f"Square: {res['square']}")
        print(f"Symmetric: {res['symmetric']}")
        if res['square'] and res['symmetric']:
            print(f"min_eig = {res['min_eig']:.6e}")
            if res['spd']:
                print("=> Matrix is symmetric positive definite (SPD).")
            elif res['psd']:
                print("=> Matrix is symmetric positive semidefinite (PSD), may be singular.")
            else:
                print("=> Matrix is symmetric but indefinite (has negative eigenvalue).")
        else:
            print("=> SPD/PSD classification not applicable.")

        print(f'\n')

        # Read solution vector u (skip '%' and blank lines)
        u_mfem = read_mfem_vector(u_txt_file_path)
        if u_mfem.size == 0:
            print(f"Warning: u file {u_txt_file_path} empty or no numeric tokens found; skipping example.")
            continue

        if u_mfem.size != N_inferred:
            print(f"Warning: length of u ({u_mfem.size}) != inferred N ({N_inferred}) for files {a_fname},{u_fname}. Proceeding but check data consistency.")

        # pdb.set_trace()

        
        b_mfem = A.dot(u_mfem)

        A, u_mfem, b_mfem = reorder_with_rcm(A, u_mfem)
        
        print(f'b_mfem shape: {b_mfem.shape}')
        print(f'Error in ||A*u_mfem - b_mfem||: {np.linalg.norm(b_mfem - A.dot(u_mfem))}')

        # Construct sparse coo tensor in torch
        vals_tensor = torch.tensor(vals_raw, dtype=torch.float64).to(device)
        rows_tensor = torch.tensor(rows, dtype=torch.int64).to(device)
        cols_tensor = torch.tensor(cols, dtype=torch.int64).to(device)
        A_tensor = torch.sparse_coo_tensor(indices=torch.stack([rows_tensor, cols_tensor]), values=vals_tensor, size=(N_inferred, N_inferred), dtype=torch.float64).to(device)

        results.append((A, A_tensor, u_mfem, b_mfem))

    return results
