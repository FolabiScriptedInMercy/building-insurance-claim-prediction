# building-insurance-claim-prediction
Machine learning model to predict building insurance claims using LightGBM with 70% recall and 0.80 ROC-AUC
# Building Insurance Claim Prediction Model

## 📋 Project Overview
A machine learning model to predict building insurance claims using LightGBM algorithm.

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Test F1-Score | 0.5355 |
| Test Recall | 70.3% |
| Test ROC-AUC | 0.8025 |
| Decision Threshold | 0.55 |

## 📊 Dataset

- **Samples**: 7,160 buildings
- **Features**: 13
- **Target**: Claim (0 = No Claim, 1 = Claim)
- **Class Imbalance**: 3.38:1

## 🔧 Features Used

1. YearOfObservation
2. Insured_Period
3. Residential
4. Building_Painted
5. Building_Fenced
6. Garden
7. Settlement
8. Building Dimension
9. Building_Type
10. Date_of_Occupancy
11. NumberOfWindows
12. Age_of_Building (Engineered)
13. Geo_Code_TargetEnc (Encoded)

## 🤖 Models Tested

- XGBoost
- LightGBM ✅ (Best)
- CatBoost
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Logistic Regression
- KNN
- Naive Bayes

## 📁 Repository Structure


## 🚀 How to Use

```python
import joblib

# Load model and preprocessing files
model = joblib.load('best_model_lightgbm.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
geo_encoding = joblib.load('geo_target_encoding.pkl')
threshold = joblib.load('optimal_threshold.pkl')['threshold']

# Make predictions
probability = model.predict_proba(X_scaled)[:, 1]
prediction = (probability >= threshold).astype(int)
