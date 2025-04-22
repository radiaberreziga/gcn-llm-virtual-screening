import torch
import numpy as np
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
import os

# Class for LLM model (ChemBERTa)
class LLM_Model(nn.Module):
    def __init__(self, llm_name="seyonec/ChemBERTa-zinc-base-v1"):
        super(LLM_Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModel.from_pretrained(llm_name)

    def forward(self, smiles):
        """
        Tokenize the SMILES strings and get embeddings from the model
        """
        inputs = self.tokenizer(smiles, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Get the mean of the last hidden state

# Function to load SMILES from a molecule supplier
def get_smiles_from_supplier(supplier):
    """
    Extract SMILES strings from a RDKit molecule supplier.
    """
    smiles = []
    for molecule in supplier:
        if molecule is not None:
            smile = Chem.MolToSmiles(molecule)
            smiles.append(smile)
    return smiles

# Function to save encoded SMILES embeddings
def save_embeddings(embeddings, output_path):
    """
    Save the SMILES embeddings to a file
    """
    np.save(output_path, embeddings)

# Function to encode SMILES using LLM model
def encode_smiles(smiles, llm_model):
    """
    Encode a list of SMILES strings into embeddings using the LLM model
    """
    encoded_smiles = []
    for i, smile in enumerate(smiles):
        print(f"Encoding SMILES {i+1}/{len(smiles)}")
        encoded_smiles.append(llm_model(smile))
    return encoded_smiles

# Main function to load molecules, process them and save embeddings
def main(supplier, output_dir, batch_size=1000):
    """
    Main processing function to handle SMILES extraction, encoding and saving embeddings
    """
    llm_model = LLM_Model(llm_name="seyonec/ChemBERTa-zinc-base-v1")  # Initialize the LLM model

    # Step 1: Extract SMILES from supplier
    print("Extracting SMILES from molecules...")
    smiles = get_smiles_from_supplier(supplier)

    # Step 2: Encode SMILES into LLM embeddings
    print(f"Encoding {len(smiles)} SMILES strings...")
    encoded_smiles = encode_smiles(smiles, llm_model)

    # Step 3: Save the embeddings to the output directory
    output_path = os.path.join(output_dir, 'smile_encode4000-all.npy')
    print(f"Saving embeddings to {output_path}...")
    save_embeddings(encoded_smiles, output_path)
    print("Embeddings saved successfully!")

# Example usage (uncomment and replace with actual molecule supplier)
# from rdkit.Chem import SDMolSupplier
# supplier = SDMolSupplier('/path/to/your/molecule_file.sdf')

# output_dir = '/content/drive/MyDrive/drug_sdf/data_encoded/your_sup_name'
# main(supplier, output_dir)
