import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Preprocessing function (if needed)
def preprocess_data(X, y):
    # Ensure data is in numpy or pandas form
    X = np.array(X)
    y = np.array(y)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the pipeline to include scaling and the SVM classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling the features
    ('svm', SVC(kernel='rbf'))     # Initialize SVM with RBF kernel
])

# Define parameter grid for GridSearchCV
param_grid = {
    'svm__C': [0.1, 1, 10, 100],           # Regularization parameter C
    'svm__gamma': ['scale', 0.1, 1, 10],   # Gamma parameter for RBF kernel
    'svm__class_weight': [None, 'balanced'] # Class weight (for handling imbalanced classes)
}

# Main function for training and evaluation
def train_and_evaluate(X, y):
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Initialize GridSearchCV with the pipeline and parameter grid
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)
    
    # Retrieve the best model from GridSearchCV
    best_svm_model = grid_search.best_estimator_
    
    # Train the final model on the full training set
    best_svm_model.fit(X_train, y_train)
