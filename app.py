import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="GP Predictive Intelligence | BUET",
    page_icon="💹",
    layout="wide"
)

# --- Professional Academic Styling ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 24px; color: #0f172a; font-weight: 600; }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #64748b;
        text-align: center;
        font-size: 12px;
        border-top: 1px solid #e2e8f0;
        padding: 8px 0;
        z-index: 100;
    }
    .result-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .label-desc {
        font-size: 12px;
        color: #64748b;
        margin-top: -10px;
        margin-bottom: 10px;
        display: block;
    }
    </style>
    <div class="footer">
        Developed by Fahmida Supta | Department of ICT, BUET
    </div>
    """, unsafe_allow_html=True)

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    files = ['gp_rf_model.joblib', 'scaler.joblib', 'X_valid.joblib', 'Y_valid.joblib']
    if not all(os.path.exists(f) for f in files):
        st.error("⚠️ System Error: Model assets not found.")
        st.stop()
    try:
        model, scaler = joblib.load('gp_rf_model.joblib'), joblib.load('scaler.joblib')
        xv, yv = joblib.load('X_valid.joblib'), joblib.load('Y_valid.joblib')
        preds = model.predict(xv)
        r2 = r2_score(yv, preds) * 100
        hr = np.mean((np.diff(np.array(yv).flatten()) > 0) == (np.diff(np.array(preds).flatten()) > 0)) * 100
        return model, scaler, hr, r2
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        st.stop()

model, scaler, hit_ratio, r2_val = load_assets()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Model Reliability")
    st.metric("Directional Hit Ratio", f"{hit_ratio:.2f}%")
    st.metric("R² Variance Score", f"{r2_val:.2f}%")
    st.divider()
    st.caption("Algorithm: Random Forest Ensemble")

# --- Dashboard ---
st.title("💹 Grameenphone Predictive Intelligence")
st.markdown("##### AI-Driven Forecasting for DSE Opening Prices")
st.markdown("---")

# Main Input Section
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Price Points**")
    prev_open = st.number_input("Prev Open", value=246.0) # Added to calculate O-C
    
    h = st.number_input("Prev High", value=250.0)
    st.markdown('<span class="label-desc">ℹ️ Highest price yesterday</span>', unsafe_allow_html=True)
    
    l = st.number_input("Prev Low", value=245.0)
    st.markdown('<span class="label-desc">ℹ️ Lowest price yesterday</span>', unsafe_allow_html=True)
    
    c = st.number_input("Prev Close", value=248.0)
    st.markdown('<span class="label-desc">ℹ️ Final price yesterday</span>', unsafe_allow_html=True)

with col2:
    st.write("**Technical Indicators**")
    v = st.number_input("Prev Volume", value=1000000)
    st.markdown('<span class="label-desc">ℹ️ Total turnover yesterday</span>', unsafe_allow_html=True)
    
    ma = st.number_input("Prev MA20", value=246.0)
    st.markdown('<span class="label-desc">ℹ️ 20-day trendline</span>', unsafe_allow_html=True)
    
    rsi = st.number_input("Prev RSI14", value=55.0)
    st.markdown('<span class="label-desc">ℹ️ Momentum gauge</span>', unsafe_allow_html=True)

with col3:
    st.write("**Calculated Features (Auto)**")
    
    # AUTOMATIC CALCULATIONS
    calc_oc = prev_open - c
    calc_lh = l - h
    
    st.metric("Auto Prev (Open-Close)", f"{calc_oc:.2f}")
    st.markdown('<span class="label-desc">ℹ️ Candle Body: Open minus Close</span>', unsafe_allow_html=True)
    
    st.metric("Auto Prev (Low-High)", f"{calc_lh:.2f}")
    st.markdown('<span class="label-desc">ℹ️ Candle Wick: Low minus High</span>', unsafe_allow_html=True)

# Execution Logic
st.markdown("##")
if st.button("EXECUTE FORECAST ENGINE", use_container_width=True):
    # Vector transformation using calculated fields
    features = np.array([[h, l, c, v, ma, rsi, calc_oc, calc_lh]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #64748b; font-size: 14px; text-transform: uppercase;">Predicted Opening Price</span>
                <h1 style="margin-top: 8px; color: #0f172a;">{prediction:.2f} BDT</h1>
                <p style="color: #22c55e; font-size: 14px; margin-top: 10px;">✔ Dynamic forecast generated successfully.</p>
            </div>
        """, unsafe_allow_html=True)
    with res_col2:
        st.latex(r"\hat{y}_{t+1} = \text{RF}(\mathbf{x}_{t})")
        st.info("The model has automatically computed the price spreads to ensure high inference precision.")
