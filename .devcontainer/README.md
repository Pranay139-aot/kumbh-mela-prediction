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
├── app.py                             # Streamlit App (Main interactive web application & Streamlit Cloud entry point)
├── data_preparation_and_training.py  # Dataset compilation, feature scaling, model CV & evaluation script
├── predict_api.py                     # Flask REST API exposing POST /predict & GET /health
├── index.html                         # Functional HTML interface
├── kumbh_crowd_risk_dataset.csv       # Compiled dataset (Sourced + Synthetic)
├── kumbh_risk_model_pipeline.joblib   # Serialized trained scikit-learn/XGBoost pipeline
├── requirements.txt                   # Required Python libraries (includes Streamlit)
└── README.md                          # Project documentation (this file)
```

---

## 5. How to Run & Deploy on Streamlit

### Option A: Local Streamlit Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.

### Option B: Deploying on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **"New app"**, select your GitHub repository, set **Main file path** to `app.py`.
4. Click **Deploy!** Streamlit Cloud will automatically install `requirements.txt`, run `app.py`, build the dataset/model pipeline on startup if missing, and serve the application online.

---

## 6. Running via Flask REST API Backend (Optional)

If you prefer running as a Flask REST API backend:
1. Execute model training:
   ```bash
   python data_preparation_and_training.py
   ```
2. Start Flask server:
   ```bash
   python predict_api.py
   ```
3. Test `POST /predict` endpoint via `http://127.0.0.1:5000/predict` or view `index.html`.

---

## 7. Academic Credits & Project Details
- **Course**: 3rd Year B.Tech AIML (Machine Learning Mini-Project)
- **Domain**: Urban Risk Assessment & Mass Gathering Management
- **Target Event**: Nashik-Trimbakeshwar Simhastha Kumbh Mela 2027
