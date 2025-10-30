import streamlit as st
import pandas as pd
import joblib

# ==== Title ====
st.title("📞 Telco Customer Churn Prediction App")

st.write("Predict whether a customer will churn (leave) based on their service details.")

# ==== Load model and scaler ====
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model_XGBoost.pkl') 
# ==== Input features ====
st.subheader("🔹 Enter Customer Info:")

tenure = st.number_input("Customer Tenure (months):", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($):", min_value=10.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges ($):", min_value=0.0, max_value=10000.0, value=800.0)
contract = st.selectbox("Contract Type:", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service:", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method:", ["Electronic check", "Mailed check", "Credit card", "Bank transfer"])

# ==== Convert to DataFrame ====
input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'InternetService': [internet_service],
    'PaymentMethod': [payment_method]
})

# ===== Preprocessing (Dummy Encoding) =====
input_encoded = pd.get_dummies(input_df)
# Adjust columns to match training model
train_columns = model.get_booster().feature_names if hasattr(model, "get_booster") else None

if train_columns:
    for col in train_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[train_columns]

# ==== Scale ====
X_scaled = scaler.transform(input_encoded)

# ==== Predict ====
if st.button("Predict Churn"):
    prob = model.predict_proba(X_scaled)[0,1]
    churn = (prob > 0.5)
    st.write("### 🧾 Probability of Churn:", f"{prob:.2f}")
    if churn:
        st.error("⚠️ This customer is likely to CHURN!")
    else:
        st.success("✅ This customer is likely to stay.")
