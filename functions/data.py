from glob import glob
import os
import numpy as np
import torch

from functions.utils import ddtype

class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, n, graph=True, size=None) -> None:
        super().__init__()
        self.folder = folder
        self.files = os.listdir(folder)
        self.graph = graph
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        g = torch.load(os.path.join(self.folder, self.files[idx]))
        g.x = g.x.to(ddtype)
        g.edge_attr = g.edge_attr.to(ddtype)
        g.s = g.s.to(ddtype)
        g.b = g.b.to(ddtype)
        
        return g