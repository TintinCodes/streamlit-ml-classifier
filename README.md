# Bank Marketing Campaign Prediction – Machine Learning Classification

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on demographic, financial, and campaign-related attributes.

Six classification algorithms are implemented and evaluated using standard performance metrics. The best-performing model is deployed through a Streamlit web application for interactive prediction.

### Objectives
- Implement 6 classification models
- Evaluate using Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Compare model performance
- Deploy using Streamlit Community Cloud

---

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Type:** Binary Classification  

### Dataset Characteristics

- Total Instances: 550 (balanced sample)
- Total Features: 16
- Target Variable: Term Deposit Subscription (0 = No, 1 = Yes)
- Class Distribution: 275 per class (balanced)
- Train-Test Split: 80% training, 20% testing (stratified)

The dataset contains information collected during direct marketing campaigns of a Portuguese banking institution.

### Preprocessing Steps

- Label Encoding for categorical features
- Feature Scaling using StandardScaler
- Stratified Train-Test Split
- No missing values in dataset

---

## 3. Models Implemented

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## 4. Model Evaluation

### Evaluation Metrics Used

- Accuracy  
- AUC (ROC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Performance Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.7273 | 0.8284 | 0.7119 | 0.7636 | 0.7368 | 0.4558 |
| Decision Tree | 0.7364 | 0.7364 | 0.7321 | 0.7455 | 0.7387 | 0.4728 |
| kNN | 0.6909 | 0.7375 | 0.6981 | 0.6727 | 0.6852 | 0.3821 |
| Naive Bayes | 0.5273 | 0.7405 | 0.514 | 1.0 | 0.679 | 0.1674 |
| Random Forest | 0.7909 | 0.8817 | 0.75 | 0.8727 | 0.8067 | 0.5898 |
| XGBoost | 0.8091 | 0.8916 | 0.7742 | 0.8727 | 0.8205 | 0.6233 |

---

## 5. Model Observations

- **Logistic Regression:** Strong baseline with balanced precision and recall. Good interpretability.
- **Decision Tree:** Slightly improved performance with interpretable decision rules.
- **kNN:** Lower performance due to high-dimensional feature space sensitivity.
- **Naive Bayes:** High recall but poor overall predictive quality (low MCC).
- **Random Forest:** Strong ensemble performance with high recall and robustness.
- **XGBoost:** Best overall performer with highest Accuracy, AUC, and MCC.

Ensemble models outperform individual models, indicating the presence of complex feature interactions.

---

## 6. Project Structure

streamlit-ml-classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── model/
├── train_models.py
├── download_dataset.py
├── bank_marketing.csv
├── model_results.csv
├── trained_models.pkl
├── scaler.pkl
└── test_data.csv



---

## 7. Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/TintinCodes/streamlit-ml-classifier.git
cd streamlit-ml-classifier
```

### Step 2: Install Dependencies
```
pip install -r requirements.txt
```

### Step 3: Training Models
```
python model/train_models.py
```

### Step 4: Run Streamlit App
```
streamlit run app.py
```

### Step 2: Install Dependencies
```
pip install -r requirements.txt
```

## 8. Streamlit Application Features
    - CSV Upload for predictions
    - Model selection dropdown
    - Display of evaluation metrics
    - Confusion matrix
    - Classification report
    - Model comparison charts

Live App: https://app-ml-classifier-juke4xjyfacnjncun3kbrp.streamlit.app/


## 9. Technologies Used
    - Python
    - scikit-learn
    - XGBoost
    - Streamlit
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn