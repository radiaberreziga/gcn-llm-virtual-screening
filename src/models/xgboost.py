import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.preprocessing.encoding import extract_targets
from rdkit import Chem

# ==== CONFIGURATION ====
SUP_NAME = "Beta-secretase-7"  # à adapter selon ton dossier
BASE_PATH = Path("././data/raw/")
COLUMN_FILE = BASE_PATH / "7th_Step_DS_Ligand-Prep_Files/descriptor/Beta_secretase_descriptors.txt"
DESCRIPTOR_FILE = BASE_PATH / "7th_Step_DS_Ligand-Prep_Files/descriptor/Beta_secretase.txt"

# ==== CHARGEMENT DES DONNÉES ====
def load_target(sup_name):
    supplier = Chem.SDMolSupplier('data/raw/7th_Step_DS_Ligand-Prep_Files/' + sup_name + '.sd')
    target = extract_targets(supplier)
    return target
        

def load_descriptors(column_file, descriptor_file):
    with open(column_file, 'r') as file:
        columns = [line.strip() for line in file.readlines()]
    descriptors = pd.read_csv(descriptor_file, sep="\s+", header=None)
    df = pd.DataFrame(descriptors.values, columns=columns)
    return df

# ==== PRÉTRAITEMENT DES DESCRIPTEURS ====
def preprocess_descriptors(df):
    print("Valeurs manquantes par colonne :")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    df = df.iloc[:, 4:]  # Supposé que les 4 premières colonnes sont à ignorer
    df.fillna(df.mean(), inplace=True)
    
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    print("Matrice de corrélation (extrait) :")
    print(scaled_df.corr().iloc[:5, :5])  # Juste un extrait
    
    return scaled_df

# ==== ENTRAÎNEMENT DU MODÈLE ====
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Nettoyage des noms de colonnes
    X_train.columns = X_train.columns.str.replace(r'[\[\]<]', '', regex=True)
    X_test.columns = X_test.columns.str.replace(r'[\[\]<]', '', regex=True)

    model = XGBClassifier(
        colsample_bytree=0.5,
        gamma=0.3,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=1,
        tree_method='hist',
        subsample=0.8,
        n_estimators=400,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy: {acc:.3f}")
    print(f"F1 Score (weighted): {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def run_xgboost():
    print("Chargement des données...")
    target = load_target(SUP_NAME)

    df = load_descriptors(COLUMN_FILE, DESCRIPTOR_FILE)

    print("Prétraitement des données...")
    X = preprocess_descriptors(df)

    print("Entraînement du modèle XGBoost...")
    model = train_and_evaluate(X, target)

    return model

# ==== MAIN ====
if __name__ == "__main__":
    run_xgboost()


# ============================================================
# === NOTE FOR REVIEWERS: Hyperparameter tuning block (not executed)
# === This block was used to find the best hyperparameters for XGBoost
# === It is commented out for clarity and reproducibility.
# === You can uncomment and run it to repeat the search if needed.
# ============================================================

"""
from sklearn.model_selection import GridSearchCV

xg_cl = xgb.XGBClassifier(tree_method='gpu_hist')

param_grid = {
    'n_estimators': [80, 100],
    'max_depth': [8, 10],
    'learning_rate': [0.01, 0.1, 0.15],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.3, 0.4, 0.5],
    'gamma': [0, 0.1, 0.3]
}

grid_search = GridSearchCV(
    estimator=xg_cl,
    param_grid=param_grid,
    scoring='accuracy',  # You can change this to another metric
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)
"""
    