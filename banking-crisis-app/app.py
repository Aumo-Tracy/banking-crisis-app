import streamlit as st
import pickle
import numpy as np

# ── Load model and scaler ──────────────────────────────────
model  = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="African Banking Crisis Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────
st.title("🏦 African Banking Crisis Predictor")
st.markdown("""
This tool uses machine learning to predict the likelihood of a 
**banking crisis** based on key economic indicators.  
Built on historical data from 13 African countries (1860–2014).
""")

st.divider()

# ── Input Section ──────────────────────────────────────────
st.subheader("📊 Enter Economic Indicators")

col1, col2 = st.columns(2)

with col1:
    systemic_crisis = st.selectbox(
        "Systemic Crisis Occurring?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    exch_usd = st.number_input(
        "Exchange Rate (vs USD)",
        min_value=0.0,
        max_value=10000.0,
        value=1.5,
        step=0.1,
        help="Local currency exchange rate against US Dollar"
    )
    domestic_debt_default = st.selectbox(
        "Domestic Debt in Default?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    sovereign_debt_default = st.selectbox(
        "Sovereign External Debt Default?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    gdp_weighted_default = st.number_input(
        "GDP Weighted Default",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )

with col2:
    inflation_cpi = st.number_input(
        "Annual Inflation Rate (CPI %)",
        min_value=-50.0,
        max_value=500.0,
        value=5.0,
        step=0.5,
        help="Annual consumer price inflation percentage"
    )
    independence = st.selectbox(
        "Country is Independent?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    currency_crises = st.selectbox(
        "Currency Crisis Occurring?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    inflation_crises = st.selectbox(
        "Inflation Crisis Occurring?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.divider()

# ── Predict Button ─────────────────────────────────────────
if st.button("🔍 Predict Crisis Risk", use_container_width=True):

    # Assemble input in same order as training features
    input_data = np.array([[
        systemic_crisis,
        exch_usd,
        domestic_debt_default,
        sovereign_debt_default,
        gdp_weighted_default,
        inflation_cpi,
        independence,
        currency_crises,
        inflation_crises
    ]])

    # Scale and predict
    input_scaled  = scaler.transform(input_data)
    prediction    = model.predict(input_scaled)[0]
    probability   = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("📋 Prediction Result")

    if prediction == 1:
        st.error(f"""
        ### 🔴 HIGH CRISIS RISK
        **Crisis Probability: {probability*100:.1f}%**

        The model detects strong indicators of a potential 
        banking crisis. Immediate policy review recommended.
        """)
    else:
        st.success(f"""
        ### 🟢 LOW CRISIS RISK
        **Crisis Probability: {probability*100:.1f}%**

        Economic indicators appear stable. 
        Continue monitoring key metrics.
        """)

    # Show input summary
    with st.expander("📌 View Input Summary"):
        import pandas as pd
        summary = pd.DataFrame({
            'Indicator': [
                'Systemic Crisis', 'Exchange Rate (USD)',
                'Domestic Debt Default', 'Sovereign Debt Default',
                'GDP Weighted Default', 'Inflation CPI %',
                'Independence', 'Currency Crisis', 'Inflation Crisis'
            ],
            'Value': [
                systemic_crisis, exch_usd, domestic_debt_default,
                sovereign_debt_default, gdp_weighted_default,
                inflation_cpi, independence, currency_crises,
                inflation_crises
            ]
        })
        st.dataframe(summary, use_container_width=True)

st.divider()

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:grey; font-size:12px;'>
Built by Tracy Aumo · Africa AI Hub Capstone Project<br>
Model: Random Forest · Dataset: African Economic Crisis (Kaggle)
</div>
""", unsafe_allow_html=True)