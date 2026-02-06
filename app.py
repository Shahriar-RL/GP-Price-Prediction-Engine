import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="GP Market Intelligence",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Professional Styling ---
st.markdown("""
    <style>
    /* Main Background and Font */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    h1 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1e40af;
    }

    /* Fixed Footer - Bottom Right */
    .footer {
        position: fixed;
        right: 20px;
        bottom: 10px;
        color: #6b7280;
        text-align: right;
        font-size: 14px;
        font-weight: 500;
        z-index: 100;
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        border: none;
        height: 3em;
        font-weight: bold;
    }
    
    div.stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }
    </style>
    
    <div class="footer">
        Developed by Fahmida Supta
    </div>
    """, unsafe_allow_html=True)

# --- Asset Loading (Cached) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('gp_rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        xv = joblib.load('X_valid.joblib')
        yv = joblib.load('Y_valid.joblib')
        
        # Calculate Reliability
        preds = model.predict(xv)
        r2 = r2_score(yv, preds) * 100
        y_actual = np.array(yv).flatten()
        y_preds = np.array(preds).flatten()
        actual_diff = np.diff(y_actual) > 0
        pred_diff = np.diff(y_preds) > 0
        hr = np.mean(actual_diff == pred_diff) * 100
        
        return model, scaler, hr, r2
    except Exception as e:
        st.error(f"Critical error loading data: {e}")
        return None, None, 0, 0

model, scaler, hit_ratio, r2_val = load_assets()

# --- Sidebar Management ---
with st.sidebar:
    st.image("https://www.grameenphone.com/themes/custom/gp/logo.png", width=100) # Optional: GP Logo if URL is valid
    st.header("📈 Model Performance")
    st.metric("Directional Hit Ratio", f"{hit_ratio:.2f}%")
    st.metric("R² Variance Explained", f"{r2_val:.2f}%")
    st.divider()
    st.info("**Methodology:** Random Forest Regressor optimized for Dhaka Stock Exchange (DSE) price dynamics.")

# --- Main Dashboard ---
st.title("💹 Grameenphone (GP) Forecasting Engine")
st.markdown("##### AI-Driven Intelligence for Opening Price Prediction")
st.write("Enter the previous trading day's technical parameters to generate a forecast.")

# Input Organization
with st.container():
    st.subheader("Market Input Parameters")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        h = st.number_input("Prev High", value=250.0, step=0.1)
        ma = st.number_input("MA20", value=246.0, step=0.1)
    with c2:
        l = st.number_input("Prev Low", value=245.0, step=0.1)
        rsi = st.number_input("RSI (14)", value=55.0, step=0.1)
    with c3:
        c = st.number_input("Prev Close", value=248.0, step=0.1)
        oc = st.number_input("Open-Close", value=2.0, step=0.1)
    with c4:
        v = st.number_input("Volume", value=1000000, step=1000)
        lh = st.number_input("Low-High", value=-5.0, step=0.1)

# Execution and Result
st.markdown("---")
if st.button("RUN PREDICTION ENGINE", use_container_width=True):
    # Prepare Features
    features = np.array([[h, l, c, v, ma, rsi, oc, lh]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    # Professional Output Display
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.markdown(f"""
            <div style="background-color:#ffffff; padding:20px; border-radius:10px; border-left: 8px solid #1e3a8a; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <p style="margin:0; font-size:16px; color:#4b5563;">Predicted Opening Price</p>
                <h2 style="margin:0; color:#1e3a8a;">{prediction:.2f} BDT</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with res_col2:
        st.latex(r"y_{pred} = f(\mathbf{x})")
        st.caption("Theoretical Basis: Multi-tree ensemble regression.")
