import torch
import numpy as np
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class LLM_Model(nn.Module):
    def __init__(self, llm_name="seyonec/ChemBERTa-zinc-base-v1"):
        super(LLM_Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModel.from_pretrained(llm_name)

    def forward(self, smiles_list):
        inputs = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


def get_smiles_from_supplier(supplier):
    return [Chem.MolToSmiles(mol) for mol in supplier if mol is not None]


def encode_smiles_batch(smiles_batch, model):
    with torch.no_grad():
        return [emb.detach().numpy() for emb in model(smiles_batch)]


def save_encoded_smiles(embeddings, path):
    np.save(path, embeddings)


def load_encoded_parts(paths):
    all_embeddings = []
    for p in paths:
        emb = np.load(p, allow_pickle=True)
        all_embeddings.extend(emb)
    return all_embeddings
 
