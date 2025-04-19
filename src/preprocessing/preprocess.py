

x= [encode_molecule_as_matrix(mol) for mol in supplier]
Target =extract_targets(supplier)
edge_indices=[torch.tensor(rdmolops.GetAdjacencyMatrix(mol).nonzero()) for mol in supplier]#Chem.GetAdjacencyMatrix(mol)

import pickle

with open('/content/drive/MyDrive/drug_sdf/data_encoded/'+sup_name+'/edge_index.pkl', 'wb') as f:
    pickle.dump(edge_indices, f)
with open('/content/drive/MyDrive/drug_sdf/data_encoded/'+sup_name+'/x.pkl', 'wb') as f:
    pickle.dump(x, f)
with open('/content/drive/MyDrive/drug_sdf/data_encoded/'+sup_name+'/Target.pkl', 'wb') as f:
    pickle.dump(Target, f)
