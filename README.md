# Kumbh Mela Crowd-Risk Prediction System (Nashik 2027)

> **AIML 3rd Year Mini-Project**  
> *A Machine Learning Approach to Crowd Density & Risk Level Classification using Day-Type Urgency and Sector Access Infrastructure Features.*

---

## 1. Executive Summary & Problem Context

The **Nashik-Trimbakeshwar Simhastha Kumbh Mela 2027** is projected to witness an unprecedented attendance of **100 to 125 million pilgrims** over its ~60-day duration—a 4x to 5x increase compared to the ~25 million footfall recorded during the 2015 Nashik Simhastha (per Maharashtra Chief Minister's Office announcements).

Managing mass gathering risks across the twin sacred cities (Nashik Godavari ghats and Trimbakeshwar Kushaavarta Kund) requires proactive, data-driven planning.

### Why Exact Capacity Numbers Do Not Exist
In real-world mass gathering management, **no single official "ghat capacity" number exists** in clean time-series format. Publicly available attendance numbers are retrospective, contested, and bucketed estimates from news reports and police releases.

### Solution & Relative Feature Framing
Rather than fabricating exact headcounts or arbitrary threshold limits, this project frames crowd-risk prediction as a **multi-class classification problem** (categorized into **Low**, **Medium**, **High**, and **Extreme** risk levels). Capacity is modeled as a **relative / derived feature set** combining:
- **Sector Placement**: Mini-city zones defined under NTKMA area planning (e.g., Ram Kund, Tapovan, Sadhugram, Trimbakeshwar, Outer Transit).
- **Transit Bottlenecks**: Parking-to-ghat walking distances (km) and shuttle bus feeder routes.
- **Pedestrian Throughput Tier**: Tier 1 (High capacity / multi-bridge), Tier 2 (Moderate), Tier 3 (Constrained bottleneck).
- **Infrastructure Expansion**: Whether the sector features newly expanded bathing ghats and standing areas.

---

## 2. Dataset Construction & Sourcing Methodology

The dataset (`kumbh_crowd_risk_dataset.csv`) combines:
1. **Real Historical Sourced Benchmarks (`is_synthetic = 0`)**: Manually compiled from reported figures across past major Kumbh events:
   - *Nashik Simhastha 2015*: 1st, 2nd, and 3rd Shahi Snan days (~3.0M to 4.0M daily bathers at Ram Kund & Trimbakeshwar), opening flag hoisting, and regular midweek days.
   - *Prayagraj Kumbh 2019 & 2025*: Mauni Amavasya (~45-50M peak single-day surges), Makar Sankranti, and Paush Purnima.
   - *Haridwar Kumbh 2021*: Somvati Amavasya under restricted protocols.
   - Each real record contains source citations in `data_source_notes`.
2. **Domain-Driven Realistic Synthetic Data (`is_synthetic = 1`)**:
   - 440 synthetic samples modeling daily sector operations for Nashik 2027.
   - Generated using physics/domain-driven logic incorporating distance penalties, day-type weights, ghat expansion mitigations, and stochastic noise.
   - Explicitly tagged with `is_synthetic = 1` for complete dataset transparency.

---

## 3. Machine Learning Architecture & Feature Importance

### Models Evaluated
The system trains and compares 3 distinct classification models:
1. **Logistic Regression** (L2 Regularized Baseline)
2. **Random Forest Classifier** (100 Decision Trees, Max Depth = 8)
3. **XGBoost / Gradient Boosting Classifier**

### Training & Evaluation Methodology
- **Train/Test Split**: 80% Training / 20% Stratified Test set.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation (`StratifiedKFold`).
- **Metrics Tracked**: Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and 4x4 Confusion Matrix.
- **Overfitting Alert**: Automated check flagging any model exhibiting >95% accuracy to prevent presenting synthetic heuristic artifacts as genuine overconfidence.

### Feature Importance Takeaways
Feature importance ranking reveals:
1. **Day-Type Category** (Shahi Snan, Amavasya, Weekend) acts as the primary volume trigger (baseline surge level).
2. **Access Infrastructure Features** (Parking walk distance, Throughput tier, Ghat expansion status) act as critical moderators—determining whether a high-volume day escalates into an **Extreme** bottleneck vs a manageable **High** risk scenario.

---

## 4. Repository Structure & Deliverables

```
├── data_preparation_and_training.py  # Dataset compilation, feature scaling, model CV & evaluation script
├── predict_api.py                     # Flask REST API exposing POST /predict & GET /health
├── index.html                         # Functional backend Web UI form with dynamic result rendering
├── kumbh_crowd_risk_dataset.csv       # Compiled dataset (Sourced + Synthetic)
├── kumbh_risk_model_pipeline.joblib   # Serialized trained scikit-learn/XGBoost pipeline
├── requirements.txt                   # Required Python libraries
└── README.md                          # Project documentation (this file)
```

---

## 5. How to Run the System

### Step 1: Install Dependencies
Ensure Python 3.9+ is installed, then run:
```bash
pip install -r requirements.txt
```

### Step 2: Train & Evaluate Models
Run the training script to build `kumbh_crowd_risk_dataset.csv`, train candidate models, print 5-fold cross-validation metrics, display feature importances, and save `kumbh_risk_model_pipeline.joblib`:
```bash
python data_preparation_and_training.py
```

### Step 3: Launch the REST API Server
Start the Flask backend server:
```bash
python predict_api.py
```
The API server will start at `http://127.0.0.1:5000/`.

### Step 4: Interact via Web Interface or API

#### A. Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
Fill out the event day type, sector zone, walk/shuttle distances, throughput tier, and ghat expansion status, then click **"Predict Crowd-Risk Level"**.

#### B. Direct REST API Test (`cURL` / Postman)
Send a `POST` request to `http://127.0.0.1:5000/predict`:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "day_type": "Shahi Snan",
    "sector_zone": "Ram Kund - Panchavati",
    "parking_walk_km": 1.8,
    "shuttle_distance_km": 8.0,
    "has_expanded_ghat": 1,
    "throughput_tier": 2,
    "days_since_start": 15,
    "festival_proximity_days": 0,
    "temperature_c": 28.5
  }'
```

Sample JSON API Response:
```json
{
  "status": "success",
  "predicted_risk_level": "Extreme",
  "confidence_score": 0.8842,
  "probabilities": {
    "Low": 0.0012,
    "Medium": 0.0241,
    "High": 0.0905,
    "Extreme": 0.8842
  },
  "model_used": "XGBoost / Gradient Boosting",
  "advisory": "Shahi Snan Peak Surge: Mandatory Sector Containment active per NTKMA guidelines. Restrict cross-sector movement."
}
```

---

## 6. Academic Credits & Project Details
- **Course**: 3rd Year B.Tech AIML (Machine Learning Mini-Project)
- **Domain**: Urban Risk Assessment & Mass Gathering Management
- **Target Event**: Nashik-Trimbakeshwar Simhastha Kumbh Mela 2027
