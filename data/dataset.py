import pickle
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os

def load_pickled_data(path, name):
    with open(os.path.join(path, f"{name}/edge_index.pkl"), "rb") as f:
        edge_index = pickle.load(f)

    with open(os.path.join(path, f"{name}/x.pkl"), "rb") as f:
        x = pickle.load(f)

    with open(os.path.join(path, f"{name}/Target.pkl"), "rb") as f:
        target = pickle.load(f)

    smile_encoded = np.load(os.path.join(path, f"{name}/smile_encode.npy"), allow_pickle=True)

    return x, edge_index, target, smile_encoded


def create_graph_dataset(x_encoded, edge_index_list, targets, smiles_encoded):
    dataset = []
    for i in range(len(x_encoded)):
        graph = Data(
            x=torch.tensor(np.array(x_encoded[i]), dtype=torch.float32),
            edge_index=edge_index_list[i],
            y=torch.tensor(targets[i], dtype=torch.float32),
            smile_mol=smiles_encoded[i],
            num_nodes=x_encoded[i].shape[0]
        )
        dataset.append(graph)
    return dataset


def load_dataloaders(path, sup_name, batch_size=32, test_size=0.2):
    x, edge_index, target, smile_encoded = load_pickled_data(path, sup_name)

    dataset = create_graph_dataset(x, edge_index, target, smile_encoded)

    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
 
