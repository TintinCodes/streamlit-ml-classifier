"""
Bank Marketing Prediction - Streamlit App
Interactive web application for model demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load models and scaler
@st.cache_resource
def load_models():
    with open('model/trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return models, scaler

# Load results
@st.cache_data
def load_results():
    return pd.read_csv('model/model_results.csv')

# Main app
def main():
    st.title("🏦 Bank Marketing Campaign Prediction System")
    st.markdown("### Machine Learning Assignment 2 - BITS Pilani")
    st.markdown("**Dataset:** Bank Marketing - Predicting Term Deposit Subscription")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["Model Overview", "Make Predictions", "Model Comparison"]
    )
    
    # Load models and data
    try:
        models, scaler = load_models()
        results_df = load_results()
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run `python model/train_models.py` first.")
        st.stop()
    
    if page == "Model Overview":
        show_model_overview(results_df)
    elif page == "Make Predictions":
        show_predictions(models, scaler)
    else:
        show_comparison(results_df)

def show_model_overview(results_df):
    """Display model performance overview"""
    st.header("📈 Model Performance Overview")
    
    st.markdown("### Evaluation Metrics for All Models")
    st.dataframe(results_df, use_container_width=True)
    
    # Best model
    best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_acc = results_df['Accuracy'].max()
    best_auc = results_df['AUC'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric("Best Accuracy", f"{best_acc:.4f}")
    with col3:
        st.metric("Best AUC", f"{best_auc:.4f}")
    with col4:
        st.metric("Total Models", len(results_df))
    
    # Dataset Info
    st.markdown("### Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Dataset:** Bank Marketing Dataset
        - **Instances:** 550 (balanced)
        - **Features:** 16
        - **Target:** Term Deposit Subscription (yes/no)
        - **Source:** UCI ML Repository
        """)
    with col2:
        st.info("""
        **Key Features:**
        - Client demographics (age, job, education)
        - Financial info (balance, loan, housing)
        - Campaign data (contact, duration, campaign)
        - Economic indicators (employment rate, CPI)
        """)
    
    # Visualizations
    st.markdown("### Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.plot(x='Model', y='Accuracy', kind='bar', ax=ax, color='skyblue')
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.plot(x='Model', y='AUC', kind='bar', ax=ax, color='lightcoral')
        ax.set_title('AUC Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('AUC Score')
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        plt.tight_layout()
        st.pyplot(fig)

def show_predictions(models, scaler):
    """Make predictions using uploaded data"""
    st.header("🔮 Make Predictions")
    
    st.markdown("""
    Upload a CSV file with the following features to predict whether a client will subscribe to a term deposit:
    - age, job, marital, education, default, balance, housing, loan
    - contact, day, month, duration, campaign, pdays, previous, poutcome
    """)
    
    # Model selection
    model_name = st.selectbox(
        "Select a model:",
        list(models.keys())
    )
    
    selected_model = models[model_name]
    
    # File upload
    st.markdown("### Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file with bank marketing campaign data"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.markdown("### Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Check if target column exists
        if 'y' in df.columns:
            X = df.drop('y', axis=1)
            y = df['y']
            has_labels = True
            # Convert to binary if needed
            if y.dtype == 'object':
                y = (y == 'yes').astype(int)
        elif 'deposit' in df.columns:
            X = df.drop('deposit', axis=1)
            y = df['deposit']
            has_labels = True
            # Convert to binary if needed
            if y.dtype == 'object':
                y = (y == 'yes').astype(int)
        else:
            X = df
            has_labels = False
        
        # Make predictions
        if st.button("🚀 Run Prediction"):
            with st.spinner("Making predictions..."):
                try:
                    # Encode categorical variables (same as training)
                    from sklearn.preprocessing import LabelEncoder
                    
                    X_encoded = X.copy()
                    categorical_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
                    
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    
                    # Scale data
                    X_scaled = scaler.transform(X_encoded)
                    
                    # Predict
                    y_pred = selected_model.predict(X_scaled)
                    y_pred_proba = selected_model.predict_proba(X_scaled)[:, 1]
                    
                    # Show predictions
                    st.markdown("### Prediction Results")
                    results = pd.DataFrame({
                        'Prediction': ['Subscribe' if p == 1 else 'No Subscription' for p in y_pred],
                        'Probability': y_pred_proba,
                        'Confidence': ['High' if (p > 0.7 or p < 0.3) else 'Medium' for p in y_pred_proba]
                    })
                    st.dataframe(results, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(y_pred))
                    with col2:
                        st.metric("Predicted Subscriptions", (y_pred == 1).sum())
                    with col3:
                        st.metric("Predicted Non-Subscriptions", (y_pred == 0).sum())
                    
                    # If labels available, show metrics
                    if has_labels:
                        st.markdown("### Model Performance on Uploaded Data")
                        
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
                        with col2:
                            st.metric("Precision", f"{precision_score(y, y_pred, zero_division=0):.4f}")
                        with col3:
                            st.metric("Recall", f"{recall_score(y, y_pred, zero_division=0):.4f}")
                        with col4:
                            st.metric("F1 Score", f"{f1_score(y, y_pred, zero_division=0):.4f}")
                        
                        # Confusion Matrix
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=['No Subscribe', 'Subscribe'],
                                   yticklabels=['No Subscribe', 'Subscribe'])
                        ax.set_title(f'Confusion Matrix - {model_name}')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                        
                        # Classification Report
                        st.markdown("### Classification Report")
                        report = classification_report(y, y_pred, 
                                                      target_names=['No Subscribe', 'Subscribe'],
                                                      output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    st.error("Please ensure your CSV file has the correct format and all required features.")

def show_comparison(results_df):
    """Show detailed model comparison"""
    st.header("📊 Detailed Model Comparison")
    
    # Metrics to compare
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    # Create comparison chart
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Best performers
    st.markdown("### 🏆 Best Performers by Metric")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Model per Metric")
        best_models = {}
        for metric in metrics:
            best_idx = results_df[metric].idxmax()
            best_models[metric] = f"{results_df.loc[best_idx, 'Model']} ({results_df.loc[best_idx, metric]:.4f})"
        
        st.table(pd.DataFrame(best_models.items(), columns=['Metric', 'Best Model']))
    
    with col2:
        st.markdown("#### Overall Rankings")
        # Calculate average rank
        ranks = results_df[metrics].rank(ascending=False)
        results_df_copy = results_df.copy()
        results_df_copy['Avg Rank'] = ranks.mean(axis=1)
        rankings = results_df_copy[['Model', 'Avg Rank']].sort_values('Avg Rank')
        rankings['Rank'] = range(1, len(rankings) + 1)
        st.table(rankings[['Rank', 'Model', 'Avg Rank']])

# Run app
if __name__ == "__main__":
    main()