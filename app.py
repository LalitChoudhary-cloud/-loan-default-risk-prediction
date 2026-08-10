import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

# ---------- Load model + metadata (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, metadata, explainer

model, metadata, explainer = load_model()
FEATURE_COLUMNS = metadata["feature_columns"]
THRESHOLD = metadata["threshold"]

st.title("💳 Loan Default Risk Predictor")
st.caption(
    "Random Forest model trained on the UCI 'Default of Credit Card Clients' dataset. "
    "Enter borrower details to get a risk score, decision, and explanation."
)

# ---------- Input form ----------
st.header("Borrower Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Profile")
    limit_bal = st.number_input("Credit Limit (₹)", min_value=1000, max_value=1000000, value=50000, step=1000)
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox(
        "Education", options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}[x]
    )
    marriage = st.selectbox(
        "Marital Status", options=[1, 2, 3],
        format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x]
    )

with col2:
    st.subheader("Repayment status (past 6 months)")
    st.caption("-1/0 = paid on time, 1+ = months delayed")
    pay_1 = st.slider("Most recent month (PAY_1)", -2, 8, 0)
    pay_2 = st.slider("2 months ago (PAY_2)", -2, 8, 0)
    pay_3 = st.slider("3 months ago (PAY_3)", -2, 8, 0)
    pay_4 = st.slider("4 months ago (PAY_4)", -2, 8, 0)
    pay_5 = st.slider("5 months ago (PAY_5)", -2, 8, 0)
    pay_6 = st.slider("6 months ago (PAY_6)", -2, 8, 0)

with col3:
    st.subheader("Bill & payment amounts (₹, past 6 months)")
    bill_amts = []
    pay_amts = []
    for i in range(1, 7):
        b = st.number_input(f"Bill amount month {i}", min_value=-100000, max_value=1000000, value=10000, step=500, key=f"bill_{i}")
        bill_amts.append(b)
    for i in range(1, 7):
        p = st.number_input(f"Amount paid month {i}", min_value=0, max_value=1000000, value=2000, step=500, key=f"pay_{i}")
        pay_amts.append(p)

predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ---------- Feature engineering (must mirror the training notebook exactly) ----------
def build_features(limit_bal, sex, education, marriage, age,
                    pay_status, bill_amts, pay_amts):
    row = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
        "MARRIAGE": marriage, "AGE": age,
    }
    for i in range(6):
        row[f"PAY_{i+1}"] = pay_status[i]
        row[f"BILL_AMT{i+1}"] = bill_amts[i]
        row[f"PAY_AMT{i+1}"] = pay_amts[i]

    avg_bill = np.mean(bill_amts)
    avg_pay = np.mean(pay_amts)
    row["avg_bill_amt"] = avg_bill
    row["avg_pay_amt"] = avg_pay
    row["utilization_ratio"] = float(np.clip(avg_bill / limit_bal, -2, 5))

    monthly_ratios = []
    for b, p in zip(bill_amts, pay_amts):
        ratio = p / b if b > 0 else 1.0
        monthly_ratios.append(np.clip(ratio, 0, 3))
    row["avg_payment_ratio"] = float(np.mean(monthly_ratios))

    row["avg_delay"] = float(np.mean(pay_status))
    row["delay_worsening"] = int(pay_status[0] > pay_status[5])
    row["max_delay"] = int(np.max(pay_status))

    # age bucket one-hot (must match training: drop_first=True dropped "21-30")
    bins = metadata["age_bins"]
    labels = metadata["age_labels"]
    bucket = pd.cut([age], bins=bins, labels=labels)[0]
    for lbl in labels[1:]:  # skip first (dropped reference category)
        row[f"age_bucket_{lbl}"] = int(bucket == lbl)

    return pd.DataFrame([row])[FEATURE_COLUMNS]

# ---------- Prediction + explanation ----------
if predict_btn:
    pay_status = [pay_1, pay_2, pay_3, pay_4, pay_5, pay_6]
    X_input = build_features(limit_bal, sex, education, marriage, age, pay_status, bill_amts, pay_amts)

    prob = model.predict_proba(X_input)[0, 1]
    decision = "🔴 REJECT / HIGH RISK" if prob >= THRESHOLD else "🟢 APPROVE / LOW RISK"

    if prob >= 0.5:
        tier = "High"
    elif prob >= THRESHOLD:
        tier = "Medium"
    else:
        tier = "Low"

    st.header("Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Default Probability", f"{prob*100:.1f}%")
    r2.metric("Risk Tier", tier)
    r3.metric("Decision", decision, help=f"Threshold: {THRESHOLD} (cost-optimized, not default 0.5)")

    st.subheader("Why this prediction?")
    shap_values = explainer.shap_values(X_input)
    base_value = explainer.expected_value[1]
    exp = shap.Explanation(
        values=shap_values[0, :, 1],
        base_values=base_value,
        data=X_input.iloc[0].values,
        feature_names=X_input.columns.tolist()
    )
    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(fig)

    st.caption(
        "This is a portfolio/demo model trained on a public dataset. "
        "Not intended for real lending decisions."
    )
else:
    st.info("Fill in the borrower details above and click Predict Risk.")
