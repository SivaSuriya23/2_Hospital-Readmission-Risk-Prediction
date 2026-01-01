import gradio as gr
import joblib
import pandas as pd

# Load trained model, threshold, and feature list
model = joblib.load("readmission_xgb_recall_model.pkl")
threshold = joblib.load("decision_threshold.pkl")
features = joblib.load("model_features.pkl")

def predict_risk(
    admission_type,
    discharge_type,
    stay_days,
    lab_procedures,
    medications,
    inpatient_visits,
    emergency_visits,
    outpatient_visits
):
    # Create input dataframe in correct feature order
    data = pd.DataFrame([[
        admission_type,
        discharge_type,
        stay_days,
        lab_procedures,
        medications,
        inpatient_visits,
        emergency_visits,
        outpatient_visits
    ]], columns=features)

    # Predict probability
    prob = model.predict_proba(data)[0][1]
    risk_score = round(prob * 100, 2)

    # Threshold-based classification (recall-optimized)
    prediction = 1 if prob >= threshold else 0

    # Risk bands for clinical interpretation
    if risk_score >= 60:
        level = "High Risk"
    elif risk_score >= 30:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return (
        f"Readmission Risk Score: {risk_score}%\n"
        f"Risk Level: {level}\n"
        f"Decision Threshold Used: {threshold}"
    )

# Gradio Interface
interface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Number(label="Admission Type ID"),
        gr.Number(label="Discharge Disposition ID"),
        gr.Number(label="Length of Stay (Days)"),
        gr.Number(label="Number of Lab Procedures"),
        gr.Number(label="Number of Medications"),
        gr.Number(label="Prior Inpatient Visits"),
        gr.Number(label="Emergency Visits"),
        gr.Number(label="Outpatient Visits")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Hospital Readmission Risk Prediction",
    description=(
        "Predicts 30-day hospital readmission risk using a recall-optimized "
        "XGBoost model. Designed to minimize missed high-risk patients."
    )
)

interface.launch()
