import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle

# === Paths ===
TARGET_PATH = "your/path/to/Target.pkl"
COLUMN_FILE = "your/path/to/Beta_secretase_descriptors.txt"
DESCRIPTOR_FILE = "your/path/to/Beta_secretase.txt"

# === Data Loading Functions ===
def load_target(path):
    with open(path, 'rb') as f:
        target = pickle.load(f)
    return target

def load_descriptors(column_file, descriptor_file):
    with open(column_file, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    descriptors = pd.read_csv(descriptor_file, sep="\s+", header=None)
    df = pd.DataFrame(descriptors.values, columns=columns)
    return df

# === Preprocessing ===
def preprocess_descriptors(df):
    df = df.iloc[:, 4:]  # Drop molecule ID and other non-numeric columns
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
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
    target = load_target(TARGET_PATH)
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
