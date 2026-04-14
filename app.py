"""
Credit Card Fraud Detection — Streamlit App
==========================================
How to run:
    pip install streamlit joblib scikit-learn numpy pandas
    streamlit run app.py

Make sure fraud_model.pkl, scaler.pkl, and feature_names.pkl
are in the SAME folder as this file.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("feature_names.pkl", "rb") as feature_file:
        feature_names = pickle.load(feature_file)
    return model, scaler, feature_names

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
> **AI Mini Project** — Uses a trained Random Forest model to predict whether
> a credit card transaction is **Fraudulent** or **Legitimate**.
""")

st.divider()

# ── Check if model files exist ────────────────────────────────────────────────
model_files = ["fraud_model.pkl", "scaler.pkl", "feature_names.pkl"]
missing = [f for f in model_files if not os.path.exists(f)]

if missing:
    st.error(f"⚠️ Missing model files: {', '.join(missing)}")
    st.info("""
    **Steps to fix:**
    1. Run the Colab notebook completely
    2. Download `fraud_model.pkl`, `scaler.pkl`, `feature_names.pkl`
    3. Place them in the same folder as `app.py`
    4. Restart the Streamlit app
    """)
    st.stop()

model, scaler, feature_names = load_model()
st.success("✅ Model loaded successfully!")

# ── Sidebar — About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Dataset:** Kaggle Credit Card Fraud  
    **Model:** Random Forest Classifier  
    **Trained on:** 284,807 transactions  
    **Fraud cases:** Only 492 (0.17%)  

    ---
    **Metrics on test set:**
    - Precision: ~0.95
    - Recall:    ~0.84
    - F1-Score:  ~0.89
    - ROC-AUC:   ~0.97
    """)
    st.markdown("---")
    st.markdown("**AI Subject — Mini Project**")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction (CSV)"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Transaction
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Transaction Details")
    st.info("The V1–V28 features are PCA-transformed values from the original dataset.")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0,
                                  max_value=30000.0, value=100.0, step=0.01)
        time   = st.number_input("Time (seconds from first transaction)",
                                  min_value=0.0, max_value=200000.0, value=50000.0)

    st.markdown("**PCA Features (V1 – V14):**")
    cols_a = st.columns(7)
    v_vals = {}
    for i in range(1, 15):
        with cols_a[(i-1) % 7]:
            v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0,
                                               format="%.4f", key=f"v{i}")

    st.markdown("**PCA Features (V15 – V28):**")
    cols_b = st.columns(7)
    for i in range(15, 29):
        with cols_b[(i-15) % 7]:
            v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0,
                                               format="%.4f", key=f"v{i}")

    st.markdown("---")

    # ── Quick Test Buttons ────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧪 Load Sample FRAUD Transaction"):
            st.session_state["sample"] = "fraud"
            st.rerun()
    with col_b:
        if st.button("🧪 Load Sample LEGIT Transaction"):
            st.session_state["sample"] = "legit"
            st.rerun()

    # ── Predict Button ────────────────────────────────────────────────────────
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        # Build input — order must match feature_names
        input_dict = {f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)}
        input_dict["Amount_Scaled"] = (amount - 88.35) / 250.12  # approximate scaling
        input_dict["Time_Scaled"]   = (time   - 94813) / 47488

        # Reorder to match training feature order
        input_row = [input_dict.get(f, 0.0) for f in feature_names]
        input_array = np.array(input_row).reshape(1, -1)

        prediction   = model.predict(input_array)[0]
        probability  = model.predict_proba(input_array)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.error("🚨 **FRAUDULENT TRANSACTION DETECTED!**")
            st.metric("Fraud Probability",
                      f"{probability[1]*100:.2f}%",
                      delta="High Risk", delta_color="inverse")
        else:
            st.success("✅ **LEGITIMATE TRANSACTION**")
            st.metric("Legitimate Probability",
                      f"{probability[0]*100:.2f}%",
                      delta="Low Risk")

        # Probability bar
        st.markdown("**Confidence Breakdown:**")
        prob_df = pd.DataFrame({
            "Class": ["Legitimate", "Fraudulent"],
            "Probability": [probability[0], probability[1]]
        })
        st.bar_chart(prob_df.set_index("Class"))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload a CSV File for Batch Prediction")
    st.info("Upload a CSV with the same columns as the training dataset (V1–V28, Amount, Time).")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"**Uploaded:** {batch_df.shape[0]} rows, {batch_df.shape[1]} columns")
        st.dataframe(batch_df.head(5))

        if st.button("🔍 Run Batch Prediction", type="primary"):
            try:
                # Scale Amount & Time if present
                if "Amount" in batch_df.columns:
                    batch_df["Amount_Scaled"] = (batch_df["Amount"] - 88.35) / 250.12
                if "Time" in batch_df.columns:
                    batch_df["Time_Scaled"] = (batch_df["Time"] - 94813) / 47488

                # Drop original cols if present
                batch_input = batch_df[[f for f in feature_names if f in batch_df.columns]]

                preds = model.predict(batch_input)
                probs = model.predict_proba(batch_input)[:, 1]

                batch_df["Prediction"] = ["FRAUD 🚨" if p == 1 else "Legit ✅" for p in preds]
                batch_df["Fraud Probability"] = (probs * 100).round(2)

                st.success("✅ Predictions complete!")

                fraud_count = (preds == 1).sum()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", len(preds))
                col2.metric("Fraudulent", fraud_count)
                col3.metric("Legitimate", len(preds) - fraud_count)

                st.dataframe(batch_df[["Prediction", "Fraud Probability"]].head(50))

                # Download results
                csv_result = batch_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv_result,
                                   file_name="fraud_predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Make sure your CSV has columns matching V1–V28 and Amount/Time.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("💳 Credit Card Fraud Detection | AI Subject Mini Project | Built with Streamlit")
