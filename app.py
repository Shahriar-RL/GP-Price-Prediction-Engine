import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="GP Market Analysis | BUET Research",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Academic Styling ---
st.markdown("""
    <style>
    /* Professional Sidebar */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    
    /* Elegant Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Clean Footer Style */
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
        padding: 5px 0;
        z-index: 100;
    }
    
    /* Academic Body Font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Result Box Enhancement */
    .result-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    </style>
    
    <div class="footer">
        Developed by Fahmida Supta | Department of ICT, BUET
    </div>
    """, unsafe_allow_html=True)

# --- Load Analytical Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('gp_rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        xv = joblib.load('X_valid.joblib')
        yv = joblib.load('Y_valid.joblib')
        
        # Calculate Reliability Profile
        preds = model.predict(xv)
        r2 = r2_score(yv, preds) * 100
        actual_diff = np.diff(np.array(yv).flatten()) > 0
        pred_diff = np.diff(np.array(preds).flatten()) > 0
        hr = np.mean(actual_diff == pred_diff) * 100
        return model, scaler, hr, r2
    except Exception as e:
        st.error(f"Asset Loading Error: {e}")
        return None, None, 0, 0

model, scaler, hit_ratio, r2_val = load_assets()

# --- Analytical Sidebar ---
with st.sidebar:
    st.markdown("### Model Validation")
    st.metric("Directional Hit Ratio", f"{hit_ratio:.2f}%")
    st.metric("R² Score (Variance)", f"{r2_val:.2f}%")
    st.divider()
    st.markdown("""
    **Objective:** Short-term opening price forecasting for Grameenphone (GP) scrip on the Dhaka Stock Exchange.
    
    **Algorithm:** Random Forest Regressor (Ensemble Learning).
    """)

# --- Main Research Dashboard ---
st.title("💹 Grameenphone Predictive Intelligence")
st.markdown("##### Technical Parameter Input for Next-Day Opening Price Forecast")

# Organized Technical Inputs
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    h = st.number_input("Prev. High (BDT)", value=250.0, step=0.1)
    ma = st.number_input("MA20 Filter", value=246.0, step=0.1)
with col2:
    l = st.number_input("Prev. Low (BDT)", value=245.0, step=0.1)
    rsi = st.number_input("RSI (14-Day)", value=55.0, step=0.1)
with col3:
    c = st.number_input("Prev. Close (BDT)", value=248.0, step=0.1)
    oc = st.number_input("Spread (O-C)", value=2.0, step=0.1)
with col4:
    v = st.number_input("Trade Volume", value=1000000, step=1000)
    lh = st.number_input("Range (L-H)", value=-5.0, step=0.1)

# Execution Logic
st.markdown("##")
if st.button("RUN FORECAST ENGINE", use_container_width=True):
    # Vector transformation
    features = np.array([[h, l, c, v, ma, rsi, oc, lh]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    # Professional Output Presentation
    st.markdown("---")
    res_col1, res_col2 = st.columns([3, 2])
    
    with res_col1:
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #64748b; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Opening Price</span>
                <h1 style="margin-top: 8px; color: #0f172a;">{prediction:.2f} BDT</h1>
                <p style="color: #22c55e; font-size: 14px; margin-top: 10px;">✔ Model inference successfully generated.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with res_col2:
        st.info("**Analysis:** Prediction based on multivariate technical signals. Validated against historical DSE volatility.")
        st.latex(r"\hat{y}_{t+1} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}_t)")
