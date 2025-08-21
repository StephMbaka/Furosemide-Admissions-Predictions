# Exploring the relationship between furosemide dose and hospital admissions in patients with heart failure
A project using statistical and machine learning methods to predict hospital admission in heart failure patients with a focus on furosemide dosing and model interpretability using SHAP.
# Hospital Admission Risk Prediction (Heart Failure & Furosemide)

This repository contains the code and analysis for a machine learning project that predicts hospital admission risk among patients with heart failure.  
The study examines the impact of **furosemide dosage** and other clinical/demographic factors, and uses **SHAP (Shapley Additive Explanations)** to improve interpretability.

## 📌 Project Overview
- **Goal:** Predict risk of hospital admission in heart failure patients.  
- **Key Features:** Medication dose (furosemide), demographics (age, sex, BMI, IMD), comorbidities, smoking, alcohol.  
- **Methods:** Logistic Regression and XGBoost with 5-fold cross-validation.  
- **Explainability:** Odds Ratios, AUC and SHAP values to interpret feature contributions at both patient and population level.  

## 📂 Repository Structure
- `data_preprocessing/` → Jupyter notebooks for cleaning and merging raw datasets.  
- `models/` → Training scripts and cross-validation setup.  
- `shap_analysis/` → SHAP value calculation and plots for interpretability.  
- `results/` → Key outputs, tables, and figures.  
- `README.md` → Project description and usage instructions.  

## ⚙️ Requirements
- Python 3.9+  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
