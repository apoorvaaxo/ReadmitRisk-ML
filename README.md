# ReadmitRisk: Predicting 30-Day Hospital Readmissions Using Machine Learning

A machine learning project that predicts whether a diabetic patient will be readmitted to the hospital within 30 days of discharge, based on clinical and demographic data.

---

## 📊 Dataset

- Source: [UCI Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- 100K+ anonymized records from 130 hospitals
- Target: `readmitted` (Yes = <30 days, No = >30 or never)

---

## 🔧 Tech Stack

- Python, Pandas, NumPy
- scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Google Colab

---

## 📈 What I Did

- Cleaned and preprocessed real-world medical data (handled nulls, encoded categorical variables)
- Handled severe class imbalance using **SMOTE** and `class_weight='balanced'`
- Trained and evaluated models:
  - Logistic Regression
  - Random Forest
  - XGBoost (best performance)
- Improved recall for high-risk class from **2% to 49%**
- Visualized feature importance (age, lab procedures, insulin levels)

---

## 📉 Key Metrics

| Model | Accuracy | Recall (Class 1) | F1-Score (Class 1) |
|-------|----------|------------------|--------------------|
| Logistic Regression | 88.8% | 2% | 4% |
| Random Forest       | 88.8% | 3% | 5% |
| XGBoost             | 72.0% | 49% | 28% ✅

---

## 🧠 Lessons Learned

- How to build ML pipelines with real medical datasets
- How to handle class imbalance with SMOTE
- How to evaluate beyond accuracy (focus on recall for healthcare)
- How to communicate insights using visualizations

---

## 🚀 Future Work

- Add Streamlit app for prediction
- Try SHAP for model explainability
