import numpy as np
import json

def advection_data_path(path,dataset):
    train_path = f"{path}/{dataset}/train/"
    return train_path