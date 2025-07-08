# app.py
import streamlit as st
import pandas as pd
import joblib
import json

# Load model & features
model = joblib.load("settlement_strategy_model.pkl")
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# UI
st.title(" Claims Settlement Optimization System")
st.markdown("Enter claim details to receive an optimized strategy recommendation based on profit, legal costs, and customer satisfaction.")

# User Inputs from the csv
claim_amount = st.number_input("Claim Amount", min_value=0)
policy_limit = st.number_input("Policy Limit", min_value=0)
customer_tenure = st.number_input("Customer Tenure (years)", min_value=0)
customer_score = st.slider("Customer Score", 0, 100)
legal_cost_predicted = st.number_input("Estimated Legal Cost", min_value=0)
claim_type = st.selectbox("Claim Type", ["Auto", "Home", "Health"])
region = st.selectbox("Region", ["North", "South", "East", "West"])

if st.button("🔎 Optimize Settlement Strategy"):

    input_df = pd.DataFrame([{
        "Claim Amount": claim_amount,
        "Policy Limit": policy_limit,
        "Customer Tenure": customer_tenure,
        "Customer Score": customer_score,
        "Legal Cost Predicted": legal_cost_predicted,
        "Claim Type": claim_type,
        "Region": region
    }])

    #Ensure theinput matches models expected features
    input_encoded = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    # Predict probabilities
    probs = model.predict_proba(input_encoded)[0]
    actions = ["Litigate", "Mediate", "Settle"]

    # Cost & profit estimates
    cost_map = {"Litigate": legal_cost_predicted, "Mediate": 3000, "Settle": 1500}
    recovery_map = {
        "Litigate": 0.7 * policy_limit - legal_cost_predicted,
        "Mediate": 0.85 * policy_limit - 3000,
        "Settle": 0.95 * policy_limit - 1500
    }
    satisfaction_map = {"Litigate": 0.3, "Mediate": 0.6, "Settle": 0.9}

    # Calculate utility
    results = []
    for i, action in enumerate(actions):
        profit = recovery_map[action]
        satisfaction = satisfaction_map[action]
        prob = probs[i]
        utility = (0.8 * profit) + (0.2 * satisfaction * 1000)
        results.append({
            "Action": action,
            "Probability": f"{prob*100:.2f}%",
            "Expected Profit": f"${profit:,.2f}",
            "Satisfaction": f"{satisfaction*100:.0f}%",
            "Utility Score": utility
        })

    best_action = max(results, key=lambda x: x["Utility Score"])

    st.subheader("✅ Recommended Strategy")
    st.write(f"**{best_action['Action']}** (Highest Utility Score)")

    st.subheader("📊 Strategy Comparison")
    st.dataframe(pd.DataFrame(results))
