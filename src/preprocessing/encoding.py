from sklearn.preprocessing import OneHotEncoder
from src.config import load_model_params
import numpy as np
import torch



categorical_features  = ["ElementSymbol","HybridizationState","FormalCharge","Chirality"]
numerical_features    = ['AtomicNumber', 'Valence' , 'Degree', 'TotalNumHs', 'NumNeighbors', 'ExplicitValence', 'NumExplicitHs', 'NumImplicitHs']
binary_features       = ["Aromaticity","RingMembership"]

def extract_atom_features(mol):
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features = {
            "ElementSymbol": atom.GetSymbol(),
            "HybridizationState": atom.GetHybridization(),
            "FormalCharge": atom.GetFormalCharge(),
            "Chirality": atom.GetChiralTag(),
            "Aromaticity": atom.GetIsAromatic(),
            "RingMembership": atom.IsInRing(),
            "AtomicNumber": atom.GetAtomicNum(),
            "Valence": atom.GetTotalValence(),
            "Degree": atom.GetTotalDegree(),
            "TotalNumHs": atom.GetTotalNumHs(),
            "NumNeighbors": len(atom.GetNeighbors()),
            "ExplicitValence": atom.GetExplicitValence(),
            "NumExplicitHs": atom.GetNumExplicitHs(),
            "NumImplicitHs": atom.GetNumImplicitHs(),
        }
        atom_features_list.append(atom_features)
    return atom_features_list

def extract_atom_all_features(supplier):
    atom_features = {key: [] for key in categorical_features + numerical_features + binary_features}
    for mol in supplier:
        if mol is not None:
            feature_mol = extract_atom_features(mol)
            for feature_atom in feature_mol:
                for key, feature in feature_atom.items():
                    if feature not in atom_features[key]:
                        atom_features[key].append(feature)
        else:
            print("Failed to read molecule")
    return atom_features

def fit_encoder(existing_categories):
    encoder = OneHotEncoder(sparse_output=False)
    categories_reshaped = np.array([[str(category)] for category in existing_categories])
    encoder.fit(categories_reshaped)
    return encoder, encoder.categories_[0]

def transform_with_encoder(encoder, categories):
    if not isinstance(categories, list):
        categories = [categories]
    categories_reshaped = np.array([[str(category)] for category in categories])
    encoded_data = encoder.transform(categories_reshaped)
    return encoded_data[0] if len(encoded_data) == 1 else encoded_data

def fit_categorical_encoders(atom_features):
    encoders = {}
    for key in categorical_features:
        encoders[key], _ = fit_encoder(atom_features[key])
    return encoders

def encode_molecule(molecule, encoders, atom_all_features):
    mol_features = extract_atom_features(molecule)
    encoded_mol_features = []
    for atom_features in mol_features:
        encoded_atom = {}
        for key, feature in atom_features.items():
            if key in categorical_features:
                encoded_atom[key] = transform_with_encoder(encoders[key], feature)
            elif key in numerical_features:
                min_val, max_val = min(atom_all_features[key]), max(atom_all_features[key])
                encoded_atom[key] = (feature - min_val) / (max_val - min_val + 1e-9)
            else:
                encoded_atom[key] = int(bool(feature))
        encoded_mol_features.append(encoded_atom)
    return encoded_mol_features

def encode_molecule_as_matrix(mol, encoders, atom_features):
    encoded_mol = encode_molecule(mol, encoders, atom_features)
    all_arrays = []
    for atom_feature in encoded_mol:
        array = np.hstack([
            np.array(value) if not isinstance(value, np.ndarray) else value
            for value in atom_feature.values()
        ])
        all_arrays.append(array)
    return np.array(all_arrays)



def extract_targets(supplier, activity_property='Activity_Status', active_value='Active'):
    """
    Extrait la target (1 pour actif, 0 pour inactif) de chaque molécule du supplier
    en fonction d'une propriété donnée (par défaut 'Activity_Status').

    Args:
        supplier (rdkit.Chem.SDMolSupplier): Liste des molécules.
        activity_property (str): Nom de la propriété à lire sur chaque molécule.
        active_value (str): Valeur correspondant à l'activité (label 1).

    Returns:
        List[int]: Liste des targets (0 ou 1).
    """
    targets = []
    for mol in supplier:
        if mol is None:
            continue  # Skip les molécules mal chargées

        status = mol.GetProp(activity_property)
        targets.append(1 if status == active_value else 0)

    return torch.tensor(targets, dtype=torch.long)

 
