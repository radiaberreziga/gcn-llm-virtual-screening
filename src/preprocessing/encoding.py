from src.preprocessing.features_extraction import extract_atom_features
from src.config import categorical_features, numerical_features, binary_features


def extract_atom_all_features(supplier=supplier):
  atom_features = {
    "ElementSymbol": [],
    "HybridizationState": [],
    "FormalCharge": [],
    "Aromaticity": [],
    "RingMembership": [],
    "Chirality": [],

    "AtomicNumber": [],
    "Valence": [],
    "Degree": [],
    "TotalNumHs": [],
    "NumNeighbors": [],
    "ExplicitValence": [],
    "NumExplicitHs": [],
    "NumImplicitHs": []
    }

  #supplier = Chem.SDMolSupplier('/content/dataset/erB1-prepared.sd')
  for mol in supplier:
    if mol is not None:  # Check if the molecule was read successfully
      feature_mol = extract_atom_features(mol)
      for feature_atom in feature_mol:
        for key,feature in feature_atom.items():
          if (feature not in atom_features[key]):
            atom_features[key].append(feature)
    else:
        print("Failed to read molecule")


  return atom_features

def fit_encoder(existing_categories):
    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Changed line

    if(type(existing_categories[0]) == str):
      # Reshape the data to a 2D array (required by scikit-learn)
      categories_reshaped = np.array([[str(category)] for category in existing_categories])
    else:
      categories_reshaped = np.array([[category] for category in existing_categories])

    # Fit the encoder on the existing categories
    encoder.fit(categories_reshaped)

    # Store the fitted encoder and its categories
    fitted_encoder = encoder
    fitted_categories = encoder.categories_[0]

    return fitted_encoder, fitted_categories

def transform_with_encoder(encoder, categories):
    # Transform the data using the provided encoder

    if(type(categories)!=list):
      categories1 = [categories]
    if(type(categories1[0]) == str):
      categories_reshaped = np.array([[str(category)] for category in categories1])
    else:
      categories_reshaped = np.array([[category] for category in categories1])

    encoded_data = encoder.transform(categories_reshaped)

    if(type(categories)!=list):
      encoded_data = encoded_data[0]

    return encoded_data


def fit_categorical_encoders(atom_features):
  categorical_encoders = {}
  for key,features in atom_features.items():
    if(key in categorical_features):
      existing_categories = atom_features[key]
      categorical_encoders[key],_= fit_encoder(existing_categories)
  return categorical_encoders

def encode_molecule(molecule,encoders,atom_all_features,categorical_features=categorical_features,numerical_features=numerical_features):
  mol_features = extract_atom_features(molecule)
  encoded_mol_features=[]
  for atom_features in mol_features:
    encoded_atom_features={}
    for key,feature in atom_features.items():
      if(key in categorical_features):
        encoded_atom_features[key] = transform_with_encoder(encoders[key], feature)
      elif (key in numerical_features) :
        min1 = min(atom_all_features[key])
        max1 = max(atom_all_features[key])
        encoded_atom_features[key] = (feature-min1)/(max1-min1)
      else:
        if(feature==True):
          encoded_atom_features[key]=1
        else:
          encoded_atom_features[key]=0
    encoded_mol_features.append(encoded_atom_features)

  return encoded_mol_features




def encode_molecule_as_matrix(mol):
  atom_features = extract_atom_all_features()
  categorical_encoders = fit_categorical_encoders(atom_features)
  encoded_mol = encode_molecule(mol,categorical_encoders,atom_features)


  # Concatenate all values into a single NumPy array
  all_arrays = []
  for atom_feature in encoded_mol:
     array = np.hstack([np.array(value) if not isinstance(value, np.ndarray) else value for value in atom_feature.values()])
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

    return targets

 
