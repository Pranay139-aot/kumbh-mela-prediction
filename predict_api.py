"""
Kumbh Mela Crowd-Risk Prediction REST API
==========================================
Course: ML Mini-Project (3rd Year AIML)
Backend API for Nashik-Trimbakeshwar Simhastha 2027 Crowd-Risk System

Exposes:
- POST /predict: Predicts crowd-risk level & confidence score given day-type and access features.
- GET /health:   Model status, loaded schema, and metadata.
- GET /:         Serves the functional index.html interface.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

MODEL_ARTIFACT_FILE = "kumbh_risk_model_pipeline.joblib"

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Load model pipeline on startup
if not os.path.exists(MODEL_ARTIFACT_FILE):
    raise FileNotFoundError(
        f"Model file '{MODEL_ARTIFACT_FILE}' not found! Run 'data_preparation_and_training.py' first."
    )

print(f"Loading trained ML model pipeline from '{MODEL_ARTIFACT_FILE}'...")
artifact = joblib.load(MODEL_ARTIFACT_FILE)
model_name = artifact["model_name"]
pipeline = artifact["pipeline"]
risk_labels = artifact["risk_labels"]
cat_features = artifact["cat_features"]
num_features = artifact["num_features"]
results_summary = artifact.get("results_summary", {})

print(f"Loaded '{model_name}' model pipeline successfully!")


def generate_infrastructure_advisory(data, predicted_risk):
    """
    Generates actionable crowd-management advisory based on relative infrastructure features.
    """
    advisories = []
    
    walk_km = float(data.get("parking_walk_km", 1.5))
    tier = int(data.get("throughput_tier", 2))
    has_expanded = int(data.get("has_expanded_ghat", 0))
    shuttle_km = float(data.get("shuttle_distance_km", 8.0))
    day_type = data.get("day_type", "Normal Day")
    sector = data.get("sector_zone", "Ram Kund - Panchavati")

    if predicted_risk in ["Extreme", "High"]:
        if tier == 3:
            advisories.append("Critical Choke Point: Throughput is bottlenecked (Tier 3). Immediately activate new Godavari pedestrian bridges and stagger pilgrim holding lanes.")
        if walk_km > 2.0:
            advisories.append(f"Long Walk Corridor ({walk_km} km): Pilgrim fatigue and crowding likely along transit corridors. Deploy additional shuttle buses to outer parking.")
        if has_expanded == 0:
            advisories.append("Unexpanded Ghat Area: Bathing capacity is constrained. Divert crowd surges towards expanded holding sectors like Tapovan or outer ghats.")
        if day_type in ["Shahi Snan", "Amavasya"]:
            advisories.append(f"{day_type} Peak Surge: Mandatory Sector Containment active per NTKMA guidelines. Restrict cross-sector movement.")
    elif predicted_risk == "Medium":
        if walk_km > 1.8:
            advisories.append("Moderate crowd buildup expected near parking feeder gates. Ensure continuous shuttle frequency.")
        else:
            advisories.append("Normal surge managed effectively under current infrastructure capacity.")
    else:
        advisories.append("Crowd density within optimal safety thresholds. Regular sector monitoring recommended.")

    return " | ".join(advisories)


@app.route("/", methods=["GET"])
def serve_ui():
    """Serves the main functional HTML interface."""
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Returns model health and performance metrics."""
    return jsonify({
        "status": "healthy",
        "active_model": model_name,
        "supported_risk_classes": risk_labels,
        "categorical_features": cat_features,
        "numerical_features": num_features,
        "model_performance_summary": results_summary
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts crowd-risk category and confidence distribution.
    Input JSON Payload format:
    {
        "day_type": "Shahi Snan",
        "sector_zone": "Ram Kund - Panchavati",
        "parking_walk_km": 1.5,
        "shuttle_distance_km": 8.0,
        "has_expanded_ghat": 1,
        "throughput_tier": 1,
        "days_since_start": 15,
        "festival_proximity_days": 0,
        "temperature_c": 28.5
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Extract and validate inputs with safe fallbacks
        day_type = str(data.get("day_type", "Normal Day"))
        sector_zone = str(data.get("sector_zone", "Ram Kund - Panchavati"))
        parking_walk_km = float(data.get("parking_walk_km", 1.5))
        shuttle_distance_km = float(data.get("shuttle_distance_km", 8.0))
        has_expanded_ghat = int(data.get("has_expanded_ghat", 0))
        throughput_tier = int(data.get("throughput_tier", 2))
        days_since_start = int(data.get("days_since_start", 15))
        festival_proximity_days = int(data.get("festival_proximity_days", 0))
        temperature_c = float(data.get("temperature_c", 28.0))

        # Build feature DataFrame matching pipeline expected structure
        input_df = pd.DataFrame([{
            "day_type": day_type,
            "sector_zone": sector_zone,
            "parking_walk_km": parking_walk_km,
            "shuttle_distance_km": shuttle_distance_km,
            "has_expanded_ghat": has_expanded_ghat,
            "throughput_tier": throughput_tier,
            "days_since_start": days_since_start,
            "festival_proximity_days": festival_proximity_days,
            "temperature_c": temperature_c
        }])

        # Perform prediction
        predicted_risk = pipeline.predict(input_df)[0]
        
        # Calculate class probabilities if classifier supports predict_proba
        probabilities = {}
        confidence_score = 1.0
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_df)[0]
            classes = pipeline.classes_
            for cls, prob in zip(classes, probs):
                probabilities[cls] = round(float(prob), 4)
            confidence_score = round(float(max(probs)), 4)
        else:
            probabilities = {predicted_risk: 1.0}

        advisory = generate_infrastructure_advisory(data, predicted_risk)

        return jsonify({
            "status": "success",
            "predicted_risk_level": predicted_risk,
            "confidence_score": confidence_score,
            "probabilities": probabilities,
            "model_used": model_name,
            "input_summary": {
                "day_type": day_type,
                "sector_zone": sector_zone,
                "parking_walk_km": parking_walk_km,
                "shuttle_distance_km": shuttle_distance_km,
                "has_expanded_ghat": "Yes" if has_expanded_ghat == 1 else "No",
                "throughput_tier": f"Tier {throughput_tier}",
                "days_since_start": days_since_start,
                "festival_proximity_days": festival_proximity_days,
                "temperature_c": temperature_c
            },
            "advisory": advisory
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Kumbh Mela Crowd-Risk REST API server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
