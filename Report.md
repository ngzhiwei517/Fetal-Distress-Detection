# Fetal Distress Detection using Cardiotocography

## 1. Introduction
Fetal distress during labor is a critical medical condition requiring timely intervention.  
Cardiotocography (CTG) provides continuous monitoring of fetal heart rate and uterine contractions, but manual interpretation is subjective and expertise-dependent.

This project explores the use of machine learning to automatically classify fetal health states into **Normal**, **Suspect**, and **Pathologic** categories.

---

## 2. Dataset Description
The dataset used is the **UCI Cardiotocography Dataset**, consisting of 2,126 CTG recordings represented by extracted numerical features.

Target variable:
- Normal (1)
- Suspect (2)
- Pathologic (3)

The dataset is highly imbalanced, with Normal cases dominating (~78%), reflecting real clinical prevalence.

---

## 3. Data Exploration and Understanding
Exploratory Data Analysis was conducted following clinical CTG interpretation guidelines:
1. Baseline heart rate
2. Variability
3. Accelerations
4. Decelerations
5. Overall fetal condition

Key findings:
- Baseline heart rate alone shows limited discriminative power
- Variability-related features (ASTV, ALTV) strongly separate fetal outcomes
- Accelerations are more frequent in Normal cases
- Prolonged decelerations are strongly associated with Pathologic outcomes

These findings align closely with established clinical knowledge.

---

## 4. Data Cleaning and Preprocessing
- Removal of non-observational rows and metadata
- Conversion of numeric features stored as object types
- Handling of missing values via deletion
- Stratified train–test split to preserve class proportions
- Feature scaling using StandardScaler, applied after splitting to prevent data leakage

---

## 5. Model Development
Baseline models evaluated:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Class imbalance was addressed using:
- Class weighting for linear and tree-based models
- Sample-weighted training for XGBoost

---

## 6. Cross-Validation
Stratified 5-fold cross-validation was used to evaluate model stability.  
Balanced Accuracy and Macro F1-score were used as evaluation metrics.

XGBoost consistently outperformed other models, particularly in minority class detection.

---

## 7. Hyperparameter Tuning
GridSearchCV was applied to XGBoost to optimize model performance.  
The scoring objective prioritized **recall for the Pathologic class**, reflecting clinical risk considerations.

Key tuned parameters included:
- Number of estimators
- Tree depth
- Learning rate
- Subsampling ratios

---

## 8. Final Evaluation
The tuned XGBoost model was evaluated on a held-out test set.  
Results demonstrated strong recall for Pathologic cases and robust overall performance.

---

## 9. Conclusion
This study demonstrates that tree-based ensemble methods, particularly XGBoost, are well-suited for CTG-based fetal distress detection.  
By combining clinical insight with machine learning, the model provides reliable decision support for high-risk cases.




