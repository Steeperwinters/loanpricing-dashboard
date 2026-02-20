
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Pricing Dashboard", layout="wide")

# -----------------------------
# Load Model
# -----------------------------

try:
    model = joblib.load("loan_pricing_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model files not found.")
    st.stop()

LGD = 0.6

# -----------------------------
# Tabs
# -----------------------------

tab1, tab2 = st.tabs(["Model Dashboard", "Project Explanation"])

# ======================================================
# TAB 1 — MODEL DASHBOARD
# ======================================================

with tab1:

    st.title("AI-Based Loan Pricing Optimization")

    uploaded_file = st.file_uploader("Upload Loan Applicant CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Basic cleaning
        if 'int_rate' in df.columns:
            df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').astype(float)

        df = pd.get_dummies(df)

        # Ensure required columns exist
        expected_cols = scaler.feature_names_in_

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]

        # Scale
        scaled = scaler.transform(df)

        # Predict PD
        df['Predicted_PD'] = model.predict_proba(scaled)[:,1]

        # Expected Loss
        df['Expected_Loss'] = df['Predicted_PD'] * LGD * df['loan_amnt']

        # -------------------------
        # Pricing Optimization
        # -------------------------

        rates = np.arange(0.08, 0.26, 0.01)
        profits = []

        for r in rates:
            acceptance = np.exp(-0.15 * r)
            profit = acceptance * (
                (r * df['loan_amnt'] * (1 - df['Predicted_PD'])) -
                (df['Predicted_PD'] * LGD * df['loan_amnt'])
            )
            profits.append(profit.mean())

        optimal_rate = rates[np.argmax(profits)]

        # -------------------------
        # KPIs
        # -------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric("Average PD", round(df['Predicted_PD'].mean(),4))
        col2.metric("Average Expected Loss", round(df['Expected_Loss'].mean(),2))
        col3.metric("Optimal Interest Rate", str(round(optimal_rate*100,2)) + "%")

        # -------------------------
        # Profit Chart
        # -------------------------

        st.subheader("Profit vs Interest Rate")

        fig, ax = plt.subplots()
        ax.plot(rates*100, profits)
        ax.axvline(optimal_rate*100, linestyle="--")
        ax.set_xlabel("Interest Rate (%)")
        ax.set_ylabel("Average Expected Profit")

        st.pyplot(fig)

        st.subheader("Prediction Sample")
        st.dataframe(df.head())

# ======================================================
# TAB 2 — EXPLANATION
# ======================================================

with tab2:

    st.title("Project Explanation")

    st.markdown("""
This project develops an AI-based loan pricing optimization framework integrating predictive credit risk modeling with profit maximization.

The first stage of the model estimates Probability of Default using supervised machine learning trained on historical LendingClub loan data. Features such as income, debt-to-income ratio, loan amount, and credit grade are used to predict borrower-level default probability. The final model was selected based on ROC-AUC performance.

In the second stage, Expected Loss is computed using the credit risk formula PD × LGD × Loan Amount. Loan pricing is then simulated across multiple interest rates. Borrower acceptance probability is incorporated using an exponential elasticity function. The optimal rate is determined by maximizing expected profit after adjusting for both credit losses and borrower demand sensitivity.

This framework demonstrates how predictive analytics can be directly integrated into risk-adjusted pricing decisions in retail banking.
""")
