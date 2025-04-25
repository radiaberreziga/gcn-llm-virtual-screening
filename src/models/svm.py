import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle
from src.preprocessing.encoding import extract_targets
from rdkit import Chem

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
    with open(column_file, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    descriptors = pd.read_csv(descriptor_file, sep="\s+", header=None)
    df = pd.DataFrame(descriptors.values, columns=columns)
    return df

# === Preprocessing ===
def preprocess_descriptors(df):
      # Affichage des valeurs manquantes
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values[missing_values > 0])

    # Suppression des colonnes non nécessaires (ex: ID molécules, etc.)
    df = df.iloc[:, 4:]

    # Remplissage des NaN dans les colonnes numériques uniquement
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Vérification des NaN restants (dans les colonnes non numériques par ex.)
    if df.isnull().sum().any():
        print("Warning: Some NaN values still remain in non-numeric columns.")
        # Option : remplacer par 0 (ou 'missing' si catégorielles)
        df.fillna(0, inplace=True)

    # Normalisation des données
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Affichage de la matrice de corrélation
    correlation_matrix = scaled_df.corr()
    print("Correlation Matrix:\n", correlation_matrix)

    return scaled_df

# === Model Training & Evaluation ===
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

   # Clean column names
    X_train.columns = X_train.columns.str.replace('[\[\]<]', '', regex=True)
    X_test.columns = X_test.columns.str.replace('[\[\]<]', '', regex=True)

    svm_clf = SVC(kernel='rbf', C=10, gamma='scale')  # RBF kernel is common
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return svm_clf

# === Function to call from main ===
def run_svm():
    print("Loading data...")
    target = load_target(SUP_NAME)
    df = load_descriptors(COLUMN_FILE, DESCRIPTOR_FILE)

    print("Preprocessing descriptors...")
    X = preprocess_descriptors(df)

    print("Training and evaluating SVM model...")
    model = train_and_evaluate(X, target)

    return model

# === MAIN ===
if __name__ == "__main__":
    run_svm()

# ============================================================
# === NOTE FOR REVIEWERS: Hyperparameter tuning block (not executed)
# === This block was used to find the best parameters for SVM
# === You can uncomment and run it to repeat the search if needed.
# ============================================================

"""
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring='accuracy',
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
