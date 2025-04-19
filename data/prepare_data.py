# data/prepare_data.py

from rdkit import Chem
import os
import pickle

def load_sdf_supplier(folder_path, sdf_name):
   
    full_path = os.path.join(folder_path, f"{sdf_name}.sd")
    supplier = Chem.SDMolSupplier(full_path)
    return supplier

def list_sdf_files(folder_path):
    
    return [f for f in os.listdir(folder_path) if f.endswith(".sd")]
