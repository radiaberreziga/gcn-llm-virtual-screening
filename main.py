import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from src.preprocessing.preprocess import get_data
from src.config import load_model_params
from src.models.gcn_llm_model import GCN_LLM
from src.models.gcn_model import GCN

from src.preprocessing.prepare_data import prepare_dataloaders


# Chargement du dataset
sup_name="Cannabinoid_CB1_receptor-7"
supplier = Chem.SDMolSupplier('data/raw/7th_Step_DS_Ligand-Prep_Files/'+sup_name+'.sd')
model_params = load_model_params('data/raw/model_graph_params.json')

#define device 
device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
print(device)

#load data
train_dataset, test_dataset = prepare_dataloaders(supplier,sup_name)
print(train_dataset[0].smile_mol.shape[1])

#Hybrid model GCN-LLM
model = GCN_LLM(num_node_features=train_dataset[0].x.shape[1],num_classes=model_params['num_classe']
            ,hidden_channels=model_params['hidden_channels'],smile_llm_dim=train_dataset[0].smile_mol.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= model_params['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

#model GCN
model = GCN(num_node_features=train_dataset[0].x.shape[1],num_classes=model_params['num_classe']
            ,hidden_channels=model_params['hidden_channels']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= model_params['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()




print(len(train_dataset))
