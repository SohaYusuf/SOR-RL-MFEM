import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
    
name = 'A_144_p1'
txt_file_path = f'plot_data/{name}.txt'
#  Read sparse matrix entries from the text file
# N=256
sparse_matrix_entries_ = []
with open(txt_file_path, 'r') as file:
    for line in file:
        row, col, val = map(float, line.split())
        sparse_matrix_entries_.append((int(row), int(col), val))
# Extract row indices, column indices, and values
rows, cols, vals = zip(*sparse_matrix_entries_)

# Construct scipy sparse matrix
A = coo_matrix((vals, (rows, cols)), dtype=np.float64)

print(f'A: No. of entries: {len(vals)}')

plt.figure(figsize=(6, 6))
plt.spy(A, markersize=0.4)                         # marker style is default for sparse :contentReference[oaicite:3]{index=3}
plt.title(f"A shape: {A.shape}, No. of entries: {len(vals)}")
plt.xlabel("Column index")
plt.ylabel("Row index")
plt.savefig(f'{name}_spy.png')
plt.show()