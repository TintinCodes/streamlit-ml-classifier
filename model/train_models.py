"""
Bank Marketing Classification - Model Training
This script trains 6 classification models and saves evaluation metrics
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
def load_data():
    """Load the bank marketing dataset"""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(SCRIPT_DIR, 'bank_marketing.csv'),  # Same directory as script
        'model/bank_marketing.csv',  # Relative from project root
        'bank_marketing.csv',  # Current directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            df = pd.read_csv(path)
            return df
    
    # If not found, print error
    print("ERROR: Dataset not found!")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    raise FileNotFoundError("bank_marketing.csv not found")

# Preprocess data
def preprocess_data(df):
    """Prepare data for training"""
    
    # The target column is 'deposit' - rename to 'y' for consistency
    if 'deposit' in df.columns:
        df = df.rename(columns={'deposit': 'y'})
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Convert target to binary if it's yes/no
    if y.dtype == 'object':
        y = (y == 'yes').astype(int)
    
    # Check class distribution BEFORE encoding
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"Class 0: {(y==0).sum()} instances")
    print(f"Class 1: {(y==1).sum()} instances")
    
    # If severely imbalanced, we need to handle it
    if (y==0).sum() < 10 or (y==1).sum() < 10:
        print("\nWARNING: Severely imbalanced dataset!")
        print("Adjusting strategy...")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Total features: {len(X.columns)}")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
    
    # Split data (80-20 split)
    # Use stratify only if we have enough samples of each class
    min_class_count = min((y==0).sum(), (y==1).sum())
    
    if min_class_count >= 5:  # Need at least 5 samples per class for stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("WARNING: Cannot stratify - too few samples in minority class")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
    
    print(f"\nTraining set: {X_train.shape[0]} instances")
    print(f"Test set: {X_test.shape[0]} instances")
    print(f"Training set class distribution:")
    print(f"  Class 0: {(y_train==0).sum()}")
    print(f"  Class 1: {(y_train==1).sum()}")
    print(f"Test set class distribution:")
    print(f"  Class 0: {(y_test==0).sum()}")
    print(f"  Class 1: {(y_test==1).sum()}")
    
    # Verify we have both classes in training set
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Training set only has one class! Please check your dataset. Unique values: {np.unique(y_train)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_encoded.columns.tolist()

# Evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    """Calculate all evaluation metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    return metrics, y_pred

# Train all models
def train_all_models():
    """Train all 6 classification models"""
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining models...")
    print("="*50)
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_test, y_test, name)
            results.append(metrics)
            
            # Save trained model
            trained_models[name] = model
            
            print(f"{name} - Accuracy: {metrics['Accuracy']}, AUC: {metrics['AUC']}")
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Determine output directory (same as script location)
    output_dir = SCRIPT_DIR
    
    # Save results
    results_path = os.path.join(output_dir, 'model_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Save models
    models_path = os.path.join(output_dir, 'trained_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(trained_models, f)
    print(f"✓ Models saved to {models_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save test data for Streamlit app
    test_data_path = os.path.join(output_dir, 'test_data.csv')
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df['y'] = y_test.values
    X_test_df.to_csv(test_data_path, index=False)
    print(f"✓ Test data saved to {test_data_path}")
    
    return results_df

if __name__ == "__main__":
    print("="*60)
    print("Bank Marketing Classification - Model Training")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {SCRIPT_DIR}")
    
    results = train_all_models()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    print("\n✓ Training complete!")