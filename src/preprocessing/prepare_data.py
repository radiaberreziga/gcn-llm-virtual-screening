# data/prepare_data.py

from rdkit import Chem
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Dataset,Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from src.config import load_model_params
from src.preprocessing.preprocess import get_data


# Define a custom Dataset class
def data_graph_list(x_encoded, edge_index_g, target,smiles):
 return [
    Data(
        x=torch.tensor(np.array(x_encoded[i])),
        edge_index=edge_index_g[i],
        y=target[i],
        smile_mol = smiles[i],
        num_nodes=x_encoded[i].shape[0]  # Set the num_nodes attribute
    )
    for i in range(len(x_encoded))
]

def prepare_dataloaders(supplier,sup_name):
  
    model_params = load_model_params('././data/raw/model_graph_params.json')
    # Initialize dataset
    x, target, edge_indices, smile_encoded = get_data(supplier,sup_name)
    dataset = data_graph_list(x, edge_indices,target, smile_encoded)
    # Split dataset indices into training and testing
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    # Subset the dataset using the split indices
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], shuffle=False)

    return train_loader, test_loader