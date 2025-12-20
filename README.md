# 🩺 Fetal Distress Detection using Cardiotocography (CTG)

## Project Overview
This project builds a machine learning system to classify fetal health states (**Normal**, **Suspect**, **Pathologic**) from cardiotocography (CTG) recordings.  
Developed for the **MLDA@EEE Datathon 2025**, the goal is to support clinicians in early detection of fetal distress during labor.

---

## Dataset
- **Source:** [UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography)  
- **Size:** 2,126 CTG recordings  
- **Classes:** Normal (78%), Suspect (14%), Pathologic (8%)  
- Strong class imbalance typical of real-world medical data

---

## Approach
- Clinically guided exploratory data analysis (baseline, variability, accelerations, decelerations)
- Stratified train–test split to handle class imbalance
- Baseline models: Logistic Regression, SVM, Random Forest, XGBoost
- Model selection via stratified cross-validation
- Hyperparameter tuning optimized **recall for Pathologic cases**

---

## Results
- **Best model:** XGBoost
- Strong recall for Pathologic class
- Robust performance under class imbalance
- Model behavior aligns with clinical CTG interpretation

---

## Tech Stack
Python, pandas, scikit-learn, Logistic Regression,Random Forest,SVM,XGBoost, matplotlib, seaborn

---

## How to Test the Model (Inference)

This repository includes a ready-to-use inference pipeline that allows users to test the trained model on new CTG data **without manual preprocessing**.

A separate notebook, `How_to_do_prediction.ipynb`, demonstrates the full prediction workflow.

---

###  Input Requirements
- Input must be provided as a **CSV file** in the **same format as the original CTG dataset**
- The input file **may include**:
  - Metadata columns (e.g. `FileName`, `Date`, `SegFile`)
  - Target columns (`NSP`, `CLASS`)
- **No preprocessing is required by the user**
  - Column selection, feature ordering, and scaling are handled internally

---

###  Required Files
Ensure the following files are available in the working directory(can be dowloaded from model folder) :

```text
ctg_model.joblib      # trained XGBoost model
ctg_scaler.joblib     # fitted StandardScaler
ctg_features.joblib   # feature list used during training
test_ctg_full.csv     # input CTG data (original format)
