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
    .feature-desc {
        font-size: 12px;
        color: #64748b;
        margin-top: -15px;
        margin-bottom: 15px;
        font-style: italic;
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
        st.error("⚠️ Asset Loading Error: Ensure all .joblib files are in the repository.")
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
    st.caption("Training Set: 10+ Years DSE History")

# --- Dashboard ---
st.title("💹 Grameenphone Predictive Intelligence")
st.markdown("##### AI-Driven Forecasting for DSE Opening Prices")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    h = st.number_input("Prev High", value=250.0)
    st.markdown('<p class="feature-desc">ℹ️ Highest price yesterday; defines resistance.</p>', unsafe_allow_html=True)
    ma = st.number_input("Prev MA20", value=246.0)
    st.markdown('<p class="feature-desc">ℹ️ 20-day trendline; filters daily market noise.</p>', unsafe_allow_html=True)

with col2:
    l = st.number_input("Prev Low", value=245.0)
    st.markdown('<p class="feature-desc">ℹ️ Lowest price yesterday; defines support.</p>', unsafe_allow_html=True)
    rsi = st.number_input("Prev RSI14", value=55.0)
    st.markdown('<p class="feature-desc">ℹ️ Momentum gauge; signals if move is exhausted.</p>', unsafe_allow_html=True)

with col3:
    c = st.number_input("Prev Close", value=248.0)
    st.markdown('<p class="feature-desc">ℹ️ Final price; the anchor for market sentiment.</p>', unsafe_allow_html=True)
    oc = st.number_input("Prev (Open-Close)", value=2.0)
    st.markdown('<p class="feature-desc">ℹ️ Candle Body: Difference between Open and Close.</p>', unsafe_allow_html=True)

with col4:
    v = st.number_input("Prev Volume", value=1000000)
    st.markdown('<p class="feature-desc">ℹ️ Total turnover; confirms the trend strength.</p>', unsafe_allow_html=True)
    lh = st.number_input("Prev (Low-High)", value=-5.0)
    st.markdown('<p class="feature-desc">ℹ️ Candle Wick: Total daily price range.</p>', unsafe_allow_html=True)

# Execution Logic
st.markdown("##")
if st.button("EXECUTE FORECAST ENGINE", use_container_width=True):
    features = np.array([[h, l, c, v, ma, rsi, oc, lh]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #64748b; font-size: 14px; text-transform: uppercase;">Predicted Opening Price</span>
                <h1 style="margin-top: 8px; color: #0f172a;">{prediction:.2f} BDT</h1>
                <p style="color: #22c55e; font-size: 14px; margin-top: 10px;">✔ Forecast generated with statistical confidence.</p>
            </div>
        """, unsafe_allow_html=True)
    with res_col2:
        st.latex(r"\hat{y}_{t+1} = \text{RandomForest}(\mathbf{x}_t)")
        st.info("Analysis accounts for historical volatility clusters and previous-session momentum.")
