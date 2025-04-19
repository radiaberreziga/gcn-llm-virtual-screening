from rdkit import Chem

def extract_atom_features(mol):
    features = []
    for atom in mol.GetAtoms():
        features.append({
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
        })
    return features
 
