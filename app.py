import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sklearn
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
    [data-testid="stMetricValue"] { font-size: 22px; color: #0f172a; font-weight: 600; }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #64748b;
        text-align: center;
        font-size: 11px;
        border-top: 1px solid #e2e8f0;
        padding: 5px 0;
        z-index: 100;
    }
    .label-desc {
        font-size: 11px;
        color: #64748b;
        margin-top: -5px;
        margin-bottom: 10px;
        display: block;
        font-style: italic;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    <div class="footer">
        Developed by Fahmida Supta | Department of ICT, BUET
    </div>
    """, unsafe_allow_html=True)

# --- Asset Loading with Version Check ---
@st.cache_resource
def load_assets():
    files = ['gp_rf_model.joblib', 'scaler.joblib', 'X_valid.joblib', 'Y_valid.joblib']
    
    # Check if files exist
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        st.error(f"Missing Assets: {', '.join(missing)}")
        st.stop()
        
    try:
        model = joblib.load('gp_rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        xv = joblib.load('X_valid.joblib')
        yv = joblib.load('Y_valid.joblib')
        
        # Performance Calculation
        preds = model.predict(xv)
        r2 = r2_score(yv, preds) * 100
        
        # Directional Accuracy
        y_actual = np.array(yv).flatten()
        y_preds = np.array(preds).flatten()
        hr = np.mean((np.diff(y_actual) > 0) == (np.diff(y_preds) > 0)) * 100
        
        return model, scaler, hr, r2
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        st.info(f"Deployed sklearn version: {sklearn.__version__}")
        st.stop()

model, scaler, hit_ratio, r2_val = load_assets()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Model Reliability")
    st.metric("Directional Hit Ratio", f"{hit_ratio:.2f}%")
    st.metric("R² Variance Score", f"{r2_val:.2f}%")
    st.divider()
    st.caption(f"Backend: scikit-learn v{sklearn.__version__}")

# --- Dashboard ---
st.title("💹 Grameenphone Predictive Intelligence")
st.markdown("##### AI-Driven Forecasting for DSE Opening Prices")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Core Price Points**")
    prev_open = st.number_input("Prev Open", value=246.0)
    
    h = st.number_input("Prev High", value=250.0)
    st.markdown('<span class="label-desc">ℹ️ Highest price yesterday</span>', unsafe_allow_html=True)
    
    l = st.number_input("Prev Low", value=245.0)
    st.markdown('<span class="label-desc">ℹ️ Lowest price yesterday</span>', unsafe_allow_html=True)
    
    c = st.number_input("Prev Close", value=248.0)
    st.markdown('<span class="label-desc">ℹ️ Final price yesterday</span>', unsafe_allow_html=True)

with col2:
    st.markdown("**Trend & Momentum**")
    v = st.number_input("Prev Volume", value=1000000)
    st.markdown('<span class="label-desc">ℹ️ Total turnover yesterday</span>', unsafe_allow_html=True)
    
    ma = st.number_input("Prev MA20", value=246.0)
    st.markdown('<span class="label-desc">ℹ️ 20-day trendline</span>', unsafe_allow_html=True)
    
    rsi = st.number_input("Prev RSI14", value=55.0)
    st.markdown('<span class="label-desc">ℹ️ Momentum gauge</span>', unsafe_allow_html=True)

with col3:
    st.markdown("**Automated Feature Extraction**")
    
    # Automatic logic for derived features
    calc_oc = prev_open - c
    calc_lh = l - h
    
    st.metric("Auto Prev (Open-Close)", f"{calc_oc:.2f}")
    st.markdown('<span class="label-desc">ℹ️ Candle Body calculation</span>', unsafe_allow_html=True)
    
    st.metric("Auto Prev (Low-High)", f"{calc_lh:.2f}")
    st.markdown('<span class="label-desc">ℹ️ Candle Wick calculation</span>', unsafe_allow_html=True)

# Execution Logic
st.markdown("##")
if st.button("EXECUTE FORECAST ENGINE", use_container_width=True):
    # Vector transformation
    features = np.array([[h, l, c, v, ma, rsi, calc_oc, calc_lh]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #64748b; font-size: 13px; text-transform: uppercase;">Predicted Opening Price</span>
                <h2 style="margin-top: 5px; color: #0f172a;">{prediction:.2f} BDT</h2>
                <p style="color: #22c55e; font-size: 13px;">✔ Analysis complete.</p>
            </div>
        """, unsafe_allow_html=True)
    with res_col2:
        st.latex(r"\hat{y}_{t+1} = \text{RF}(\mathbf{x}_{t})")
