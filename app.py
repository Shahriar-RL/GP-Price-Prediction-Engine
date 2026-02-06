import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="GP Predictive Intelligence",
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
    .desc-text {
        font-size: 12px;
        color: #64748b;
        margin-top: -15px;
        margin-bottom: 10px;
        font-style: italic;
    }
    </style>
    <div class="footer">
        Developed by Fahmida Supta | Department of Finance, FBS, DU
    </div>
    """, unsafe_allow_html=True)

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    files = ['gp_rf_model.joblib', 'scaler.joblib', 'X_valid.joblib', 'Y_valid.joblib']
    missing_files = [f for f in files if not os.path.exists(f)]
    if missing_files:
        st.error(f"⚠️ Critical System Error: Missing files detected: {', '.join(missing_files)}")
        st.stop()
    try:
        model, scaler = joblib.load('gp_rf_model.joblib'), joblib.load('scaler.joblib')
        xv, yv = joblib.load('X_valid.joblib'), joblib.load('Y_valid.joblib')
        preds = model.predict(xv)
        r2 = r2_score(yv, preds) * 100
        actual_diff = np.diff(np.array(yv).flatten()) > 0
        pred_diff = np.diff(np.array(preds).flatten()) > 0
        hr = np.mean(actual_diff == pred_diff) * 100
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

# Feature List Reference for Thesis Documentation
# features_list = ['Prev_High', 'Prev_Low', 'Prev_Close', 'Prev_Volume', 'Prev_MA20', 'Prev_RSI14', 'Prev_Open-Close', 'Prev_Low-High']

col1, col2, col3, col4 = st.columns(4)

with col1:
    po = st.number_input("Prev Open", value=246.0)
    st.markdown('<p class="desc-text">ℹ️ Opening price of previous session</p>', unsafe_allow_html=True)
    
    h = st.number_input("Prev High", value=250.0)
    st.markdown('<p class="desc-text">ℹ️ Highest price yesterday</p>', unsafe_allow_html=True)

with col2:
    l = st.number_input("Prev Low", value=245.0)
    st.markdown('<p class="desc-text">ℹ️ Lowest price yesterday</p>', unsafe_allow_html=True)
    
    c = st.number_input("Prev Close", value=248.0)
    st.markdown('<p class="desc-text">ℹ️ Final price yesterday</p>', unsafe_allow_html=True)

with col3:
    v = st.number_input("Prev Volume", value=1000000)
    st.markdown('<p class="desc-text">ℹ️ Total turnover yesterday</p>', unsafe_allow_html=True)
    
    ma = st.number_input("Prev MA20", value=246.0)
    st.markdown('<p class="desc-text">ℹ️ 20-day trendline</p>', unsafe_allow_html=True)

with col4:
    rsi = st.number_input("Prev RSI14", value=55.0)
    st.markdown('<p class="desc-text">ℹ️ Momentum gauge</p>', unsafe_allow_html=True)

# --- Automatic Feature Calculation ---
# Dynamically calculating the features required by the features_list
calc_oc = po - c
calc_lh = l - h

st.markdown("---")
st.markdown("### Structural Feature Extraction (Automated)")
auto_col1, auto_col2 = st.columns(2)
with auto_col1:
    st.metric("Prev (Open-Close)", f"{calc_oc:.2f}")
    st.markdown('<p class="desc-text">ℹ️ Candle Body: Difference between Open and Close</p>', unsafe_allow_html=True)
with auto_col2:
    st.metric("Prev (Low-High)", f"{calc_lh:.2f}")
    st.markdown('<p class="desc-text">ℹ️ Candle Wick: Total daily price range</p>', unsafe_allow_html=True)

# Execution Logic
st.markdown("##")
if st.button("EXECUTE FORECAST ENGINE", use_container_width=True):
    # Vector transformation must strictly match the features_list order
    # ['Prev_High', 'Prev_Low', 'Prev_Close', 'Prev_Volume', 'Prev_MA20', 'Prev_RSI14', 'Prev_Open-Close', 'Prev_Low-High']
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
                <p style="color: #22c55e; font-size: 14px; margin-top: 10px;">✔ Forecast generated with statistical confidence.</p>
            </div>
        """, unsafe_allow_html=True)
    with res_col2:
        st.latex(r"\hat{y}_{t+1} = \text{RandomForest}(\mathbf{x}_{t})")
        st.info("The model has automatically processed the OHLC spread for precision inference.")
