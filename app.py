import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(page_title="GP Price Forecast", page_icon="💹", layout="wide")

# --- Load Assets ---
@st.cache_resource
def load_data():
    model = joblib.load('gp_rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    xv = joblib.load('X_valid.joblib')
    yv = joblib.load('Y_valid.joblib')
    
    # Calculate Dynamic Performance
    preds = model.predict(xv)
    r2 = r2_score(yv, preds) * 100
    # Ensuring flatten for directional comparison
    actual_diff = np.diff(yv.values.flatten()) > 0
    pred_diff = np.diff(preds.flatten()) > 0
    hr = np.mean(actual_diff == pred_diff) * 100
    return model, scaler, hr, r2

model, scaler, hit_ratio, r2_val = load_data()

# --- UI Layout ---
st.title("💹 Grameenphone (GP) Prediction Engine")

# Sidebar for Model Reliability
st.sidebar.header("📊 Model Reliability Profile")
st.sidebar.metric("Probabilistic Hit Ratio", f"{hit_ratio:.2f}%")
st.sidebar.metric("Explanatory Power (R2)", f"{r2_val:.2f}%")
st.sidebar.write("---")
st.sidebar.info("This model utilizes a Random Forest Regressor trained on 10+ years of DSE market data.")

# Main Input Form
st.subheader("Input Technical Indicators")
col1, col2 = st.columns(2)

with col1:
    h = st.number_input("Prev High", value=250.0, help="Highest price yesterday")
    l = st.number_input("Prev Low", value=245.0, help="Lowest price yesterday")
    c = st.number_input("Prev Close", value=248.0, help="Final closing price")
    v = st.number_input("Prev Volume", value=1000000, help="Total shares traded")

with col2:
    ma = st.number_input("Prev MA20", value=246.0, help="20-day moving average")
    rsi = st.number_input("Prev RSI14", value=55.0, help="Relative Strength Index")
    oc = st.number_input("Prev (Open-Close)", value=2.0, help="Price spread")
    lh = st.number_input("Prev (Low-High)", value=-5.0, help="Intraday range")

# Prediction Execution
if st.button("🚀 Generate Opening Price Forecast", use_container_width=True):
    features = np.array([[h, l, c, v, ma, rsi, oc, lh]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    st.markdown("---")
    # Balloon pop-up removed for a cleaner professional look
    st.success(f"### Predicted NEXT DAY OPEN Price: **{prediction:.2f} BDT**")
    
    # Formula Display for academic rigor
    st.latex(r"Forecast = RF_{regressor}(\vec{x}_{scaled})")

# --- Footer with Attribution ---
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: grey;
        text-align: center;
        font-style: italic;
        padding: 10px;
        z-index: 100;
    }
    </style>
    <div class="footer">
    <p>This work is developed by Fahmida Supta</p>
    </div>
    """,
    unsafe_allow_html=True
)
