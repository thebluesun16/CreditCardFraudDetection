"""
Credit Card Fraud Detection — Demo Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳", layout="centered")

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl"), joblib.load("feature_names.pkl")

# ── Pre-filled test cases from the actual dataset ─────────────────────────────
# These are real rows from creditcard.csv (scaled values included)

LEGIT_CASE = {
    "label": "Legitimate Transaction",
    "amount": 149.62,
    "time": 0.0,
    "V1": -1.3598, "V2": -0.0728, "V3":  2.5363, "V4":  1.3782, "V5": -0.3383,
    "V6":  0.4624, "V7":  0.2396, "V8":  0.0987, "V9":  0.3638, "V10": 0.0908,
    "V11": -0.5516,"V12": -0.6178,"V13": -0.9913,"V14": -0.3112,"V15":  1.4682,
    "V16": -0.4704,"V17":  0.2080,"V18":  0.0258,"V19":  0.4040,"V20":  0.2514,
    "V21": -0.0183,"V22":  0.2778,"V23": -0.1105,"V24":  0.0669,"V25":  0.1285,
    "V26": -0.1891,"V27":  0.1336,"V28": -0.0211,
    "Amount_Scaled": 0.244, "Time_Scaled": -1.996
}

FRAUD_CASE = {
    "label": "Fraudulent Transaction",
    "amount": 1.00,
    "time": 406.0,
    "V1": -2.3122, "V2":  1.9519, "V3": -1.6096, "V4":  3.9979, "V5": -0.5221,
    "V6": -1.4265, "V7": -2.5374, "V8": -0.0700, "V9": -0.2752,"V10": -0.5754,
    "V11":  1.9781,"V12": -1.2328,"V13":  0.7802,"V14": -2.3899,"V15":  0.4456,
    "V16": -0.9334,"V17": -0.7801,"V18":  0.7501,"V19": -0.8226,"V20":  0.5382,
    "V21":  1.3458,"V22": -1.1196,"V23":  0.1750,"V24": -0.4514,"V25": -0.2372,
    "V26": -0.6385,"V27":  0.1011,"V28":  0.1045,
    "Amount_Scaled": -0.342, "Time_Scaled": -1.988
}

def predict(case, model, feature_names):
    row = [case.get(f, 0.0) for f in feature_names]
    arr = np.array(row).reshape(1, -1)
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0]
    return pred, prob

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.markdown("**AI PBL Demo** — Click a button to test the model on a real transaction from the dataset.")
st.divider()

# Check files
if not os.path.exists("fraud_model.pkl"):
    st.error("⚠️ fraud_model.pkl not found. Run the Colab notebook first and place the .pkl files here.")
    st.stop()

model, feature_names = load_model()
st.success("✅ Model loaded — Random Forest trained on 284,807 transactions")
st.divider()

# ── Two big buttons ───────────────────────────────────────────────────────────
st.subheader("🧪 Test the Model")
col1, col2 = st.columns(2)

run_legit = col1.button("✅ Test Legitimate Transaction", use_container_width=True, type="secondary")
run_fraud = col2.button("🚨 Test Fraud Transaction", use_container_width=True, type="primary")

# ── Show transaction details + result ─────────────────────────────────────────
def show_result(case):
    st.markdown("---")
    st.markdown(f"### Transaction Details")

    c1, c2, c3 = st.columns(3)
    c1.metric("Amount", f"${case['amount']:.2f}")
    c2.metric("Time (sec)", f"{case['time']:.0f}")
    c3.metric("Features", "V1 – V28 (PCA)")

    with st.expander("🔍 View all feature values (V1–V28)"):
        v_vals = {k: v for k, v in case.items() if k.startswith("V") and not k.startswith("V1") or k == "V1"}
        cols = st.columns(7)
        v_keys = [f"V{i}" for i in range(1, 29)]
        for i, key in enumerate(v_keys):
            cols[i % 7].metric(key, f"{case.get(key, 0):.4f}")

    st.markdown("### 🎯 Prediction")
    pred, prob = predict(case, model, feature_names)

    if pred == 1:
        st.error("## 🚨 FRAUD DETECTED")
        st.progress(float(prob[1]))
        cc1, cc2 = st.columns(2)
        cc1.metric("Fraud Probability",     f"{prob[1]*100:.2f}%")
        cc2.metric("Legitimate Probability", f"{prob[0]*100:.2f}%")
    else:
        st.success("## ✅ LEGITIMATE TRANSACTION")
        st.progress(float(prob[0]))
        cc1, cc2 = st.columns(2)
        cc1.metric("Legitimate Probability", f"{prob[0]*100:.2f}%")
        cc2.metric("Fraud Probability",      f"{prob[1]*100:.2f}%")

    st.caption(f"Model: Random Forest | Actual label in dataset: **{'FRAUD 🚨' if case == FRAUD_CASE else 'LEGIT ✅'}**")

if run_legit:
    show_result(LEGIT_CASE)
elif run_fraud:
    show_result(FRAUD_CASE)
else:
    st.info("👆 Click one of the buttons above to run a test prediction.")

st.divider()
st.caption("Credit Card Fraud Detection | AI Subject Mini Project | Random Forest Model")
