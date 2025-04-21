from rdkit import Chem
import torch
from rdkit.Chem import rdmolops
import numpy as np

from src.preprocessing.encoding import encode_molecule_as_matrix, extract_targets,extract_atom_all_features, fit_categorical_encoders


def get_data(supplier,sup_name):

    atom_features = extract_atom_all_features(supplier)
    categorical_encoders = fit_categorical_encoders(atom_features)

    x = [
         encode_molecule_as_matrix(mol, categorical_encoders, atom_features)
         for mol in supplier
    ]
    target = extract_targets(supplier)
    edge_indices = [
        torch.tensor(rdmolops.GetAdjacencyMatrix(mol).nonzero(), dtype=torch.long)
        for mol in supplier
    ]
    smile_encoded = np.load('././data/raw/smile_dic/'+sup_name+'/smile_encode.npy',allow_pickle=True)

    
    return x, target, edge_indices, smile_encoded
