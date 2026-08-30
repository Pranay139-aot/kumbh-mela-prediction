"""
Kumbh Mela Crowd-Risk Prediction System (Streamlit Web App)
===========================================================
Target Event: Nashik-Trimbakeshwar Simhastha Kumbh Mela 2027
Course: ML Mini-Project (3rd Year AIML)

Runs natively with Streamlit: `streamlit run app.py`
Deployable on Streamlit Community Cloud.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Kumbh Mela Crowd-Risk AI (Nashik 2027)",
    page_icon="🚩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File Constants
DATASET_FILE = "kumbh_crowd_risk_dataset.csv"
MODEL_ARTIFACT_FILE = "kumbh_risk_model_pipeline.joblib"


@st.cache_resource
def load_or_train_pipeline():
    """
    Loads pre-trained model pipeline.
    If missing (e.g. fresh Streamlit Cloud deployment), triggers automatic training.
    """
    if not os.path.exists(MODEL_ARTIFACT_FILE) or not os.path.exists(DATASET_FILE):
        st.info("⚡ Initializing system & training ML pipeline for first-time setup...")
        from data_preparation_and_training import build_kumbh_dataset, train_and_evaluate_models
        build_kumbh_dataset()
        train_and_evaluate_models()

    artifact = joblib.load(MODEL_ARTIFACT_FILE)
    return artifact


# Load Model & Metadata
artifact = load_or_train_pipeline()
model_name = artifact["model_name"]
pipeline = artifact["pipeline"]
risk_labels = artifact["risk_labels"]
inv_risk_map = artifact.get("inv_risk_map", {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"})
cat_features = artifact["cat_features"]
num_features = artifact["num_features"]
results_summary = artifact.get("results_summary", {})


def generate_infrastructure_advisory(data, predicted_risk):
    """Generates crowd safety advisory based on relative infrastructure features."""
    advisories = []
    walk_km = data["parking_walk_km"]
    tier = data["throughput_tier"]
    has_expanded = data["has_expanded_ghat"]
    day_type = data["day_type"]

    if predicted_risk in ["Extreme", "High"]:
        if tier == 3:
            advisories.append("🚨 **Bottleneck Warning (Tier 3)**: Activate new Godavari pedestrian bridges and initiate holding zone queuing.")
        if walk_km > 2.0:
            advisories.append(f"🚶 **Long Pedestrian Corridor ({walk_km} km)**: Surge in pilgrim fatigue. Increase shuttle bus frequency from outer parking hubs.")
        if has_expanded == 0:
            advisories.append("⚠️ **Unexpanded Ghat Area**: Bathing capacity constrained. Divert crowds to expanded sectors (e.g., Tapovan).")
        if day_type in ["Shahi Snan", "Amavasya"]:
            advisories.append(f"🕉️ **{day_type} Peak Surge**: Enforce NTKMA Sector Containment protocols. Restrict cross-sector transit.")
    elif predicted_risk == "Medium":
        if walk_km > 1.8:
            advisories.append("ℹ️ **Moderate Transit Density**: Maintain continuous shuttle movement to avoid parking accumulation.")
        else:
            advisories.append("✅ **Manageable Density**: Crowd influx matches current infrastructure throughput capacity.")
    else:
        advisories.append("🟢 **Optimal Safety State**: Crowd density well within safe operational limits.")

    return "\n\n".join(advisories)


# Custom Styling
st.markdown("""
    <style>
        .main-header { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0px; }
        .sub-header { font-size: 1.0rem; color: #94a3b8; margin-bottom: 20px; }
        .risk-badge { font-size: 1.8rem; font-weight: 800; padding: 10px 24px; border-radius: 8px; text-align: center; }
        .risk-Low { background-color: rgba(16, 185, 129, 0.18); color: #10b981; border: 2px solid #10b981; }
        .risk-Medium { background-color: rgba(245, 158, 11, 0.18); color: #f59e0b; border: 2px solid #f59e0b; }
        .risk-High { background-color: rgba(249, 115, 22, 0.18); color: #f97316; border: 2px solid #f97316; }
        .risk-Extreme { background-color: rgba(239, 68, 68, 0.18); color: #ef4444; border: 2px solid #ef4444; }
    </style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-header">Kumbh Mela Crowd-Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Nashik-Trimbakeshwar Simhastha 2027 — Access & Infrastructure AI Model</div>', unsafe_allow_html=True)
st.caption("🎯 **AIML 3rd Year Mini-Project** | Powered by Machine Learning Classification & Relative Capacity Framing")

st.divider()

# Sidebar: Parameter Controls
st.sidebar.header("🎛️ Event & Access Inputs")

day_type = st.sidebar.selectbox(
    "Day Type Category",
    options=["Normal Day", "Weekend", "Amavasya", "Shahi Snan"],
    index=3,
    help="Shahi Snan & Amavasya days trigger peak spiritual attendance surges."
)

sector_zone = st.sidebar.selectbox(
    "Sector Zone",
    options=[
        "Ram Kund - Panchavati",
        "Tapovan",
        "Sadhugram",
        "Trimbakeshwar Inner Ghats",
        "Outer Transit & Parking"
    ],
    index=0,
    help="Self-contained sector zone under NTKMA planning."
)

parking_walk_km = st.sidebar.slider(
    "Parking-to-Ghat Walk (km)",
    min_value=0.5,
    max_value=4.0,
    value=1.8,
    step=0.1,
    help="Distance pilgrims must walk from shuttle drops / holding points to ghats."
)

shuttle_distance_km = st.sidebar.slider(
    "Shuttle Route Distance (km)",
    min_value=1.0,
    max_value=25.0,
    value=8.0,
    step=0.5,
    help="Shuttle transit distance from outer parking hubs (e.g. Rajur Bahula / Mohagaon)."
)

has_expanded_ghat = st.sidebar.radio(
    "Sector Has Expanded Ghat Area?",
    options=[1, 0],
    format_func=lambda x: "Yes (Increased Safe Bathing/Standing Capacity)" if x == 1 else "No (Legacy / Narrow Choke Point)",
    index=0
)

throughput_tier = st.sidebar.selectbox(
    "Pedestrian Throughput Tier",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Tier 1: High Throughput (Multi-bridge Access)",
        2: "Tier 2: Moderate Throughput",
        3: "Tier 3: Bottlenecked (Single Entry / Narrow Corridor)"
    }[x],
    index=1
)

days_since_start = st.sidebar.number_input("Days Since Event Start", min_value=0, max_value=60, value=15)
festival_proximity_days = st.sidebar.number_input("Days to Peak Bathing Date", min_value=0, max_value=10, value=0)
temperature_c = st.sidebar.slider("Ambient Temperature (°C)", min_value=15.0, max_value=42.0, value=28.5, step=0.5)

# Build Input Data Dict
input_data = {
    "day_type": day_type,
    "sector_zone": sector_zone,
    "parking_walk_km": parking_walk_km,
    "shuttle_distance_km": shuttle_distance_km,
    "has_expanded_ghat": has_expanded_ghat,
    "throughput_tier": throughput_tier,
    "days_since_start": days_since_start,
    "festival_proximity_days": festival_proximity_days,
    "temperature_c": temperature_c
}

input_df = pd.DataFrame([input_data])

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Risk Prediction",
    "📊 Model Benchmark Comparison",
    "🔍 Feature Importance",
    "📁 Dataset & Sourcing Transparency"
])

# ----------------------------------------------------
# TAB 1: Live Risk Prediction
# ----------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Input Scenario Summary")
        st.json({
            "Day Category": day_type,
            "Sector": sector_zone,
            "Walking Distance": f"{parking_walk_km} km",
            "Shuttle Route": f"{shuttle_distance_km} km",
            "Ghat Expansion": "Yes" if has_expanded_ghat == 1 else "No",
            "Throughput Tier": f"Tier {throughput_tier}",
            "Festival Proximity": f"{festival_proximity_days} days to peak"
        })

    with col_right:
        st.subheader("AI Prediction Output")
        
        pred_idx = pipeline.predict(input_df)[0]
        predicted_risk = inv_risk_map.get(pred_idx, "Medium") if isinstance(pred_idx, (int, float, np.integer)) else str(pred_idx)
        
        confidence = 1.0
        probabilities = {}
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_df)[0]
            for idx, prob in enumerate(probs):
                lbl = inv_risk_map.get(idx, f"Class_{idx}")
                probabilities[lbl] = round(float(prob), 4)
            confidence = round(float(max(probs)), 4)

        # Display Risk Badge
        st.markdown(
            f'<div class="risk-badge risk-{predicted_risk}">{predicted_risk.upper()} CROWD RISK</div>',
            unsafe_allow_html=True
        )

        st.metric(label="Model Confidence Score", value=f"{confidence * 100:.1f}%")
        st.caption(f"Evaluated using active pipeline: **{model_name}**")

    st.divider()

    # Probability Distribution Chart
    st.subheader("📈 Risk Class Probability Distribution")
    prob_df = pd.DataFrame({
        "Risk Category": ["Low", "Medium", "High", "Extreme"],
        "Probability (%)": [
            round(probabilities.get("Low", 0.0) * 100, 2),
            round(probabilities.get("Medium", 0.0) * 100, 2),
            round(probabilities.get("High", 0.0) * 100, 2),
            round(probabilities.get("Extreme", 0.0) * 100, 2)
        ]
    }).set_index("Risk Category")

    st.bar_chart(prob_df, height=220)

    # Advisory Box
    st.subheader("🛡️ Infrastructure & Crowd Management Advisory")
    advisory_text = generate_infrastructure_advisory(input_data, predicted_risk)
    st.info(advisory_text)


# ----------------------------------------------------
# TAB 2: Model Benchmark Comparison
# ----------------------------------------------------
with tab2:
    st.subheader("Model Evaluation Metrics & Cross-Validation")
    st.markdown("""
        To avoid relying on a single accuracy number on synthetic-augmented data, we evaluate models using 
        **5-Fold Stratified Cross Validation** and a **20% Holdout Test Set**.
    """)

    if results_summary:
        metrics_rows = []
        for name, metrics in results_summary.items():
            metrics_rows.append({
                "Model": name,
                "5-Fold CV Accuracy": f"{metrics.get('cv_accuracy', 0):.4f}",
                "5-Fold CV F1 (Macro)": f"{metrics.get('cv_f1', 0):.4f}",
                "Test Accuracy": f"{metrics.get('test_accuracy', 0):.4f}",
                "Test Precision (Macro)": f"{metrics.get('test_precision', 0):.4f}",
                "Test Recall (Macro)": f"{metrics.get('test_recall', 0):.4f}",
                "Test F1-Score (Macro)": f"{metrics.get('test_f1', 0):.4f}"
            })
        
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

        st.warning("""
            ⚠️ **Overfitting Check Notice**: If test accuracy approaches >95% on small datasets, it reflects 
            strong domain heuristic alignment in synthetic rules rather than flawless real-world generalization. 
            Real Kumbh operations require continuous sensor calibration.
        """)

        best_metrics = results_summary.get(model_name, {})
        if "confusion_matrix" in best_metrics:
            st.subheader(f"Confusion Matrix ({model_name})")
            cm_df = pd.DataFrame(
                best_metrics["confusion_matrix"],
                index=["Actual Low", "Actual Medium", "Actual High", "Actual Extreme"],
                columns=["Pred Low", "Pred Medium", "Pred High", "Pred Extreme"]
            )
            st.table(cm_df)


# ----------------------------------------------------
# TAB 3: Feature Importance Analysis
# ----------------------------------------------------
with tab3:
    st.subheader("Feature Importance Ranking")
    st.markdown("""
        Which factors drive crowd-risk predictions most strongly? Is it the spiritual urgency of **Day Type** 
        or the physical constraints of **Access Infrastructure**?
    """)

    classifier_step = pipeline.named_steps['classifier']
    preprocessor_step = pipeline.named_steps['preprocessor']

    cat_ohe_names = list(preprocessor_step.named_transformers_['cat'].get_feature_names_out(cat_features))
    all_feat_names = cat_ohe_names + num_features

    importances = None
    if hasattr(classifier_step, 'feature_importances_'):
        importances = classifier_step.feature_importances_
    elif hasattr(classifier_step, 'coef_'):
        importances = np.mean(np.abs(classifier_step.coef_), axis=0)

    if importances is not None:
        feat_df = pd.DataFrame({
            "Feature": all_feat_names,
            "Importance Score": importances
        }).sort_values(by="Importance Score", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"), height=380)

        st.success("""
            📌 **Key Technical Takeaway**:
            While **Day-Type** (Shahi Snan / Amavasya) acts as the primary surge volume trigger, 
            **Access Infrastructure features** (parking walk distance, throughput bottlenecks, and ghat expansion) 
            dictate whether crowd buildup escalates into **Extreme Risk** vs manageable **High Risk**.
        """)


# ----------------------------------------------------
# TAB 4: Dataset & Sourcing Transparency
# ----------------------------------------------------
with tab4:
    st.subheader("Dataset Composition & Sourcing Notes")
    st.markdown("""
        In accordance with project guidelines:
        - Real historical anchor points from past Kumbh Melas (Nashik 2015, Prayagraj 2019/2025, Haridwar 2021) 
          are tagged with `is_synthetic = 0` and cite official news/government reports.
        - Realistic domain-driven synthetic samples are tagged with `is_synthetic = 1`.
    """)

    if os.path.exists(DATASET_FILE):
        full_df = pd.read_csv(DATASET_FILE)
        st.write(f"**Total Dataset Rows**: {len(full_df)} | **Sourced Anchor Rows**: {len(full_df[full_df['is_synthetic']==0])} | **Synthetic Rows**: {len(full_df[full_df['is_synthetic']==1])}")

        st.dataframe(full_df[['date', 'event_name', 'day_type', 'sector_zone', 'parking_walk_km', 'crowd_risk_level', 'is_synthetic', 'data_source_notes']], use_container_width=True)

        csv_bytes = full_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_bytes,
            file_name="kumbh_crowd_risk_dataset.csv",
            mime="text/csv"
        )
