import json

def load_model_params(path):
    with open(path, 'r') as f:
        return json.load(f)

categorical_features  = ["ElementSymbol","HybridizationState","FormalCharge","Chirality"]
numerical_features    = ['AtomicNumber', 'Valence' , 'Degree', 'TotalNumHs', 'NumNeighbors', 'ExplicitValence', 'NumExplicitHs', 'NumImplicitHs']
binary_features       = ["Aromaticity","RingMembership"]
