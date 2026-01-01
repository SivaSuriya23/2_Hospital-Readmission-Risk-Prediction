---

# 🏥 Hospital Readmission Risk Prediction (Recall-Optimized ML)

## 📌 Project Overview

Hospital readmissions within 30 days are a critical healthcare quality metric and a major cost driver for hospitals. This project builds an **end-to-end machine learning system** to **predict 30-day hospital readmission risk**, prioritizing **high recall** to minimize missed high-risk patients.

The solution includes:

* Data preprocessing on real healthcare data
* Multiple model comparison
* Recall-optimized XGBoost model
* Explainable risk scoring
* Interactive Gradio web application
* Deployment-ready structure for Hugging Face Spaces

---

## 🎯 Business Objective

Predict whether a patient will be **readmitted within 30 days** after discharge so that hospitals can:

* Identify high-risk patients early
* Plan preventive interventions
* Reduce avoidable readmissions
* Improve care quality metrics

**Key business constraint:**
Missing a high-risk patient is more costly than raising a false alarm → **Recall is prioritized over accuracy**.

---

## 📊 Dataset Information

**Dataset Name:**
Diabetes 130-US Hospitals Dataset (1999–2008)

**Source:**
UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

**Size:**
~100,000 patient encounters across 130 US hospitals

**Target Variable:**
`readmitted`

* `<30` → Readmitted within 30 days (Positive class = 1)
* `>30`, `NO` → Not readmitted within 30 days (0)

---

## 🧠 Features Used (Patient-Friendly & Clinically Meaningful)

Only features that are realistic for nurses, clinicians, or discharge staff to provide:

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| admission_type_id        | Type of hospital admission |
| discharge_disposition_id | Discharge outcome          |
| time_in_hospital         | Length of stay (days)      |
| num_lab_procedures       | Number of lab tests        |
| num_medications          | Number of medications      |
| number_inpatient         | Prior inpatient visits     |
| number_emergency         | Prior emergency visits     |
| number_outpatient        | Prior outpatient visits    |

---

## ⚙️ Data Preprocessing

* Converted categorical readmission labels into binary target
* Handled missing values using median imputation
* Removed high-cardinality diagnosis codes for patient-friendly deployment
* Stratified train-test split to preserve class distribution

---

## 🤖 Models Evaluated

Multiple models were evaluated to ensure a fair and explainable selection:

| Model               | Purpose                      |
| ------------------- | ---------------------------- |
| Logistic Regression | Interpretable baseline       |
| Random Forest       | Non-linear benchmark         |
| XGBoost             | Final recall-optimized model |

---

## 📈 Model Comparison Results

| Model               | Accuracy | Precision | Recall    | F1 Score | ROC-AUC  |
| ------------------- | -------- | --------- | --------- | -------- | -------- |
| Logistic Regression | 0.69     | 0.17      | 0.48      | 0.25     | 0.64     |
| Random Forest       | 0.86     | 0.14      | 0.05      | 0.07     | 0.61     |
| **XGBoost (Tuned)** | 0.64     | 0.18      | **0.72+** | **0.27** | **0.68** |

**Why XGBoost?**

* Highest recall (primary healthcare metric)
* Best ROC-AUC
* Supports explainability (SHAP)
* Production-ready for tabular EHR data

---

## 🎯 Recall Optimization Strategy

To push recall above **70%**, the following techniques were applied:

* Class imbalance handling using `scale_pos_weight`
* Shallow trees with more estimators for generalization
* Lowered decision threshold (≈ 0.35 instead of 0.5)
* Probability-based risk scoring

This intentionally trades off accuracy to **minimize false negatives**, which aligns with healthcare risk management.

---

## 🧪 Explainability

* SHAP values used for global and local explanations
* Identifies key drivers such as:

  * Prior inpatient visits
  * Length of stay
  * Medication count
  * Emergency visits
* Enables clinician trust and auditability

---

## 🧮 Risk Scoring Logic

Instead of a simple binary output, the model produces a **risk score**:

```text
Risk Score = Predicted Probability × 100
```

**Risk Bands**

* **Low Risk:** < 30%
* **Medium Risk:** 30–59%
* **High Risk:** ≥ 60%

This makes the output **actionable for care teams**.

---

## 🖥️ Gradio Web Application

An interactive Gradio app allows users to:

* Enter patient admission details
* View readmission risk percentage
* See interpreted risk level
* Use recall-optimized decision logic

Designed for **Hugging Face Spaces deployment**.

---

## 🚀 Deployment (Hugging Face Spaces)

### Project Structure

```text
hospital-readmission-risk
│
├── app.py
├── readmission_xgb_recall_model.pkl
├── decision_threshold.pkl
├── model_features.pkl
├── requirements.txt
└── README.md
```

### Deployment Steps

1. Create a new Hugging Face Space
2. Select **Gradio** as the SDK
3. Upload all project files
4. App deploys automatically

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Gradio
* Joblib
* Hugging Face Spaces

---

## 📌 Key Learnings

* Accuracy is misleading for imbalanced healthcare problems
* Recall-driven evaluation is critical for patient safety
* Threshold tuning is as important as model selection
* Explainability is mandatory in healthcare ML
* Deployment considerations matter as much as modeling

---

## 📄 Resume-Ready Bullet Points

* Built a recall-optimized machine learning model to predict 30-day hospital readmissions using real US healthcare data.
* Compared multiple classification models and selected XGBoost based on recall-focused evaluation under severe class imbalance.
* Tuned decision thresholds to achieve over 70% recall, minimizing missed high-risk patients.
* Developed an explainable risk scoring system and deployed it as an interactive Gradio application on Hugging Face.

---

## 📬 Future Improvements

* Add SHAP explanations directly in the web app
* Integrate diagnosis code grouping
* Add probability calibration
* Connect to real-time EHR pipelines

---

## 👤 Author

**Siva Suriya**
Data Scientist | Machine Learning | Healthcare Analytics

---