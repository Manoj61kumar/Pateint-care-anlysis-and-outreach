from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from neo4j import GraphDatabase

# ----------------------------------
# 🔹 Load ML Models & Encoders
# ----------------------------------
try:
    disease_model = load_model('pred_model/model2/disease_prediction_model.h5')
    disease_label_encoders = joblib.load('pred_model/model2/label_encoders.pkl')
    disease_scaler = joblib.load('pred_model/model2/robust_scaler.pkl')
    mlb = joblib.load('pred_model/model2/multi_label_binarizer.pkl')

    rec_model = joblib.load('recom_model/proto_model/next_recommendation_model.pkl')
    rec_le = joblib.load('recom_model/proto_model/next_recommendation_labelencoder.pkl')
    outcome_le = joblib.load('recom_model/proto_model/outcome_labelencoder.pkl')

except Exception as e:
    print(f"❌ Error loading ML models: {e}")

# ----------------------------------
# 🔹 Neo4j Connection Handling
# ----------------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # Change this to your actual password

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
except Exception as e:
    print(f"❌ Error connecting to Neo4j: {e}")

# ----------------------------------
# 🔹 FastAPI Setup
# ----------------------------------
app = FastAPI(title="Healthcare AI Semantic Network")

# Close Neo4j Connection on Shutdown
@app.on_event("shutdown")
def shutdown():
    if driver:
        driver.close()
        print("✅ Neo4j connection closed.")

# ----------------------------------
# 🔹 Define Patient Data Schema
# ----------------------------------
class PatientInput(BaseModel):
    patient_id: str
    age: int
    gender: str
    region: str
    bmi: float
    symptoms: str  # Comma-separated symptoms
    systolic_bp: float
    diastolic_bp: float
    glucose: float
    wbc: float
    outcome: str
    # New fields to match training features:
    category: str = "cardiovascular"      # default category (adjust if needed)
    medication: str = "Lisinopril"         # default medication (adjust if needed)
    severity: int = 2                      # default severity

# ----------------------------------
# 🔹 ML Prediction Function
# ----------------------------------
def predict_disease_and_recommendation(patient: PatientInput):
    """Run ML models to predict disease and recommended action"""
    try:
        # --- Build Feature Vector for Disease Prediction ---
        # Process symptoms
        symptoms_list = patient.symptoms.split(", ")
        symptoms_encoded = pd.DataFrame(mlb.transform([symptoms_list]), columns=mlb.classes_)
        # Ensure all symptom columns exist
        missing_symptoms = np.setdiff1d(mlb.classes_, symptoms_encoded.columns).tolist()
        for col in missing_symptoms:
            symptoms_encoded[col] = 0
        # Maintain correct column order
        symptoms_encoded = symptoms_encoded[mlb.classes_]

        # Build a DataFrame with all required features (order must match training)
        # Expected non-symptom columns:
        # ['age', 'gender', 'region', 'bmi', 'category', 'severity', 'glucose', 'systolic_bp', 'diastolic_bp', 'wbc', 'medication', 'outcome']
        feature_dict = {
            "age": patient.age,
            "bmi": patient.bmi,
            "glucose": patient.glucose,
            "systolic_bp": patient.systolic_bp,
            "diastolic_bp": patient.diastolic_bp,
            "wbc": patient.wbc
        }
        # Transform categorical fields using saved label encoders
        # Note: For fields not provided by the client, we use the default values from PatientInput.
        for col in ["gender", "region", "category", "medication", "outcome"]:
            val = getattr(patient, col)
            # Transform returns an array; take the first element.
            feature_dict[col] = disease_label_encoders[col].transform([val])[0]
        # Add severity (numeric, assumed to be already in correct format)
        feature_dict["severity"] = patient.severity

        # Create DataFrame (make sure columns are in the same order as during training)
        # Adjust the order to match training: 
        # ['age', 'gender', 'region', 'bmi', 'category', 'severity', 'glucose', 'systolic_bp', 'diastolic_bp', 'wbc', 'medication', 'outcome']
        cols_order = ['age', 'gender', 'region', 'bmi', 'category', 'severity', 'glucose', 'systolic_bp', 'diastolic_bp', 'wbc', 'medication', 'outcome']
        input_df = pd.DataFrame([feature_dict], columns=cols_order)

        # Scale the numeric features (assume these are the same as used in training)
        numeric_cols = ['age', 'bmi', 'glucose', 'systolic_bp', 'diastolic_bp', 'wbc']
        input_df[numeric_cols] = disease_scaler.transform(input_df[numeric_cols])

        # Concatenate with the symptom features
        input_data_final = np.hstack((input_df.values, symptoms_encoded.values))

        # Debug: Check feature shape
        expected_features = disease_model.input_shape[-1]
        actual_features = input_data_final.shape[1]
        if actual_features != expected_features:
            raise ValueError(f"❌ Feature size mismatch! Expected {expected_features}, got {actual_features}")

        # Predict disease
        disease_pred = disease_model.predict(input_data_final)
        disease_index = np.argmax(disease_pred)
        predicted_disease = disease_label_encoders['diagnosis'].inverse_transform([disease_index])[0]

        # --- Build Feature Vector for Recommendation ---
        # Encode outcome for recommendation model
        outcome_enc = outcome_le.transform([patient.outcome])[0]
        # For recommendation, features: ['age', 'bmi', 'glucose', 'systolic_bp', 'diastolic_bp', 'wbc', 'severity', 'outcome_enc']
        rec_input = np.array([[patient.age, patient.bmi, patient.glucose,
                                patient.systolic_bp, patient.diastolic_bp, patient.wbc, patient.severity, outcome_enc]])
        # Predict next recommendation
        recommendation_pred = rec_model.predict(rec_input)
        predicted_recommendation = rec_le.inverse_transform(recommendation_pred)[0]

        return predicted_disease, predicted_recommendation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error in prediction: {e}")

# ----------------------------------
# 🔹 Store Data in Neo4j
# ----------------------------------
def store_in_neo4j(patient: PatientInput, disease: str, recommendation: str):
    """Store patient, disease, and recommendation in Neo4j"""
    try:
        with driver.session() as session:
            session.run("""
                MERGE (p:Patient {id: $patient_id})
                SET p.age = $age, p.gender = $gender, p.region = $region, p.bmi = $bmi
                
                MERGE (d:Disease {name: $disease})
                MERGE (p)-[:DIAGNOSED_WITH]->(d)

                MERGE (r:Recommendation {action: $recommendation})
                MERGE (p)-[:RECOMMENDED_NEXT]->(r)
            """, 
            patient_id=patient.patient_id, 
            age=patient.age, 
            gender=patient.gender, 
            region=patient.region, 
            bmi=patient.bmi, 
            disease=disease, 
            recommendation=recommendation)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error storing data in Neo4j: {e}")

# ----------------------------------
# 🔹 FastAPI Endpoints
# ----------------------------------
@app.post("/predict")
def predict(patient: PatientInput):
    try:
        disease, recommendation = predict_disease_and_recommendation(patient)
        store_in_neo4j(patient, disease, recommendation)

        return {
            "patient_id": patient.patient_id,
            "predicted_disease": disease,
            "next_recommendation": recommendation,
            "neo4j_browser_url": "http://localhost:7474/browser/"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------
# 🔹 Run API Server
# ----------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
