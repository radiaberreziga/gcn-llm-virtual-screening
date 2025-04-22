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
from src.train_utils import test, train
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
for step, data in enumerate(train_dataset):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)

    print()
'''
#Hybrid model GCN-LLM
model = GCN_LLM(num_node_features=train_dataset[0].x.shape[1],num_classes=model_params['num_classe']
            ,hidden_channels=model_params['hidden_channels'],smile_llm_dim=train_dataset[0].smile_mol.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= model_params['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

#model GCN
#model = GCN(num_node_features=train_dataset[0].x.shape[1],num_classes=model_params['num_classe']
 #           ,hidden_channels=model_params['hidden_channels']).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr= model_params['learning_rate'])
#criterion = torch.nn.CrossEntropyLoss()

acc_GCN=[]
fscore_GCN= []
all_labels_roc_GCN = []  # Collect true labels for ROC curve
all_probs_roc_GCN = []

for epoch in range(1,600):
    train(train_dataset, model, device, criterion, optimizer)
    train_acc,_,_,_ = test(train_dataset,model,device)
    test_acc,f1_test, test_labels, test_probs  = test(test_dataset,model,device)
    acc_GCN.append(test_acc)
    fscore_GCN.append(f1_test)
    all_labels_roc_GCN.extend(test_labels)
    all_probs_roc_GCN.extend(test_probs)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Test F1: {f1_test:.3f}')



'''

