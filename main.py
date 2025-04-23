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
from src.models.xgboost import run_xgboost
from src.models.svm import run_svm


# === CONFIGURATION ===
SUP_NAME = "Cannabinoid_CB1_receptor-7"
SD_FILE = f"data/raw/7th_Step_DS_Ligand-Prep_Files/{SUP_NAME}.sd"
MODEL_CONFIG_FILE = "data/raw/model_graph_params.json"

# === DEVICE ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === LOAD DATASET ===
supplier = Chem.SDMolSupplier(SD_FILE)
train_dataset, test_dataset = prepare_dataloaders(supplier, SUP_NAME)
model_params = load_model_params(MODEL_CONFIG_FILE)

# === DEBUGGING DATA ===
for step, data in enumerate(train_dataset):
    print(f"Step {step + 1}:")
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

# === TRAIN HYBRID MODEL (GCN + LLM) ===
model = GCN_LLM(
    num_node_features=train_dataset.dataset[0].x.shape[1],
    num_classes=model_params['num_classe'],
    hidden_channels=model_params['hidden_channels'],
    smile_llm_dim=train_dataset.dataset[0].smile_mol.shape[1]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

print("=== Training GCN + LLM model ===")
for epoch in range(1, 601):
    train(train_dataset, model, device, criterion, optimizer)
    train_acc, _, _, _ = test(train_dataset, model, device)
    test_acc, f1_test, _, _ = test(test_dataset, model, device)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Test F1: {f1_test:.3f}')

# === TRAIN BASELINE GCN MODEL ===
model = GCN(
    num_node_features=train_dataset.dataset[0].x.shape[1],
    num_classes=model_params['num_classe'],
    hidden_channels=model_params['hidden_channels']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

print("=== Training baseline GCN model ===")
for epoch in range(1, 601):
    train(train_dataset, model, device, criterion, optimizer)
    train_acc, _, _, _ = test(train_dataset, model, device)
    test_acc, f1_test, _, _ = test(test_dataset, model, device)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Test F1: {f1_test:.3f}')

# === TRADITIONAL ML MODELS ===
if __name__ == "__main__":
    print("=== Running SVM baseline ===")
    run_svm()

    print("=== Running XGBoost baseline ===")
    run_xgboost()