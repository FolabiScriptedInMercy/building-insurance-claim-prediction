import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SecureGuard Insurance | Claim Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 50%, #006699 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #00d4ff); }
        to { filter: drop-shadow(0 0 20px #00d4ff); }
    }
    
    /* Subheader */
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Company badge */
    .company-badge {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
    }
    
    /* Result cards */
    .result-claim {
        background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: shake 0.5s ease-in-out;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .result-no-claim {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: celebrate 0.5s ease-in-out;
    }
    
    @keyframes celebrate {
        0% { transform: scale(0.8); opacity: 0; }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Risk level badges */
    .risk-high {
        background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 700;
        display: inline-block;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa502 0%, #ff7f00 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 700;
        display: inline-block;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #2ed573 0%, #26de81 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a0a0a0;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #2d5a87;
    }
    
    /* Divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    model = joblib.load('best_model_lightgbm.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    geo_encoding = joblib.load('geo_target_encoding.pkl')
    threshold_data = joblib.load('optimal_threshold.pkl')
    threshold = threshold_data['threshold'] if isinstance(threshold_data, dict) else 0.55
    return model, scaler, label_encoders, geo_encoding, threshold

model, scaler, label_encoders, geo_encoding, threshold = load_model()

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<div class="company-badge">🏛️ SECUREGUARD INSURANCE</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Building Insurance Claim Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Advanced Machine Learning | Accurate Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR INPUTS
# ============================================================================
with st.sidebar:
    st.markdown("## 🏢 Building Information")
    st.markdown("---")
    
    year_of_observation = st.selectbox(
        "📅 Year of Observation",
        options=[2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        index=3
    )
    
    insured_period = st.slider(
        "📋 Insured Period",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1
    )
    
    st.markdown("---")
    st.markdown("## 🏠 Property Details")
    
    residential = st.selectbox(
        "🏘️ Residential Property",
        options=[0, 1],
        format_func=lambda x: "✅ Yes" if x == 1 else "❌ No"
    )
    
    building_painted = st.selectbox(
        "🎨 Building Painted",
        options=["V", "N"],
        format_func=lambda x: "✅ Yes" if x == "V" else "❌ No"
    )
    
    building_fenced = st.selectbox(
        "🚧 Building Fenced",
        options=["V", "N"],
        format_func=lambda x: "✅ Yes" if x == "V" else "❌ No"
    )
    
    garden = st.selectbox(
        "🌳 Has Garden",
        options=["V", "O"],
        format_func=lambda x: "✅ Yes" if x == "V" else "❌ No"
    )
    
    st.markdown("---")
    st.markdown("## 📍 Location Details")
    
    settlement = st.selectbox(
        "🏙️ Settlement Type",
        options=["U", "R"],
        format_func=lambda x: "🌆 Urban" if x == "U" else "🌾 Rural"
    )
    
    geo_code = st.text_input(
        "📍 Geo Code",
        value="6088",
        help="Enter the geographic code for the building location"
    )
    
    st.markdown("---")
    st.markdown("## 📐 Building Specifications")
    
    building_dimension = st.number_input(
        "📏 Building Dimension (sqm)",
        min_value=1,
        max_value=25000,
        value=1500
    )
    
    building_type = st.selectbox(
        "🏗️ Building Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Type {x}"
    )
    
    date_of_occupancy = st.number_input(
        "📆 Year of Occupancy",
        min_value=1800,
        max_value=2024,
        value=1990
    )
    
    number_of_windows = st.slider(
        "🪟 Number of Windows",
        min_value=1,
        max_value=10,
        value=4
    )

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_claim(input_data):
    df = input_data.copy()
    
    df['Age_of_Building'] = df['YearOfObservation'] - df['Date_of_Occupancy']
    df['Age_of_Building'] = df['Age_of_Building'].clip(lower=0)
    
    df['Geo_Code_TargetEnc'] = df['Geo_Code'].astype(str).map(geo_encoding)
    df['Geo_Code_TargetEnc'].fillna(geo_encoding.mean(), inplace=True)
    df = df.drop('Geo_Code', axis=1)
    
    for col in ['Building_Painted', 'Building_Fenced', 'Garden', 'Settlement']:
        df[col] = label_encoders[col].transform(df[col].astype(str))
    
    feature_order = [
        'YearOfObservation', 'Insured_Period', 'Residential', 
        'Building_Painted', 'Building_Fenced', 'Garden', 'Settlement',
        'Building Dimension', 'Building_Type', 'Date_of_Occupancy',
        'NumberOfWindows', 'Age_of_Building', 'Geo_Code_TargetEnc'
    ]
    df = df[feature_order]
    
    df_scaled = scaler.transform(df)
    
    probability = model.predict_proba(df_scaled)[:, 1][0]
    prediction = 1 if probability >= threshold else 0
    
    return probability, prediction

# ============================================================================
# MAIN CONTENT
# ============================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📊 Property Summary")
    
    input_data = pd.DataFrame({
        'YearOfObservation': [year_of_observation],
        'Insured_Period': [insured_period],
        'Residential': [residential],
        'Building_Painted': [building_painted],
        'Building_Fenced': [building_fenced],
        'Garden': [garden],
        'Settlement': [settlement],
        'Building Dimension': [building_dimension],
        'Building_Type': [building_type],
        'Date_of_Occupancy': [date_of_occupancy],
        'NumberOfWindows': [number_of_windows],
        'Geo_Code': [geo_code]
    })
    
    # Display summary in a nice format
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏠 Building Info</h4>
            <p>📏 Size: <strong>{building_dimension} sqm</strong></p>
            <p>🏗️ Type: <strong>{building_type}</strong></p>
            <p>🪟 Windows: <strong>{number_of_windows}</strong></p>
            <p>📆 Built: <strong>{date_of_occupancy}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📍 Location Info</h4>
            <p>🏙️ Settlement: <strong>{"Urban" if settlement == "U" else "Rural"}</strong></p>
            <p>📍 Geo Code: <strong>{geo_code}</strong></p>
            <p>🏘️ Residential: <strong>{"Yes" if residential == 1 else "No"}</strong></p>
            <p>🚧 Fenced: <strong>{"Yes" if building_fenced == "V" else "No"}</strong></p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Risk Assessment")
    
    if st.button("🔮 ANALYZE RISK", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyzing building data..."):
            import time
            time.sleep(1)  # Small delay for effect
            probability, prediction = predict_claim(input_data)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Probability display
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #00d4ff; margin-bottom: 0.5rem;">Claim Probability</h2>
            <h1 style="font-size: 4rem; color: white; margin: 0;">{probability:.1%}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(probability)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Prediction result
        if prediction == 1:
            st.markdown("""
            <div class="result-claim">
                <h2>⚠️ CLAIM LIKELY</h2>
                <p>High probability of insurance claim for this property</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-no-claim">
                <h2>✅ LOW CLAIM RISK</h2>
                <p>This property shows low probability of insurance claim</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk level badge
        st.markdown("<h3 style='text-align: center; color: white;'>Risk Classification</h3>", unsafe_allow_html=True)
        
        if probability >= 0.6:
            st.markdown('<div style="text-align: center;"><span class="risk-high">🔴 HIGH RISK</span></div>', unsafe_allow_html=True)
            st.warning("⚠️ **Recommendation:** Conduct thorough property inspection before policy approval.")
        elif probability >= 0.4:
            st.markdown('<div style="text-align: center;"><span class="risk-medium">🟡 MEDIUM RISK</span></div>', unsafe_allow_html=True)
            st.info("📋 **Recommendation:** Standard underwriting process with additional documentation.")
        else:
            st.markdown('<div style="text-align: center;"><span class="risk-low">🟢 LOW RISK</span></div>', unsafe_allow_html=True)
            st.success("✅ **Recommendation:** Eligible for streamlined approval process.")

# ============================================================================
# MODEL PERFORMANCE SECTION
# ============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📈 Model Performance Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #00d4ff;">F1-Score</h4>
        <h2 style="color: white;">0.5355</h2>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #00d4ff;">Recall</h4>
        <h2 style="color: white;">70.3%</h2>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #00d4ff;">ROC-AUC</h4>
        <h2 style="color: white;">0.8025</h2>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #00d4ff;">Threshold</h4>
        <h2 style="color: white;">55%</h2>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <h3 style="color: #00d4ff;">🏛️ SecureGuard Insurance</h3>
    <p>Powered by LightGBM Machine Learning Model</p>
    <p>Developed by <strong>FolabiScriptedInMercy</strong></p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        © 2024 SecureGuard Insurance. All rights reserved.<br>
        This tool is for assessment purposes only and should not be the sole basis for underwriting decisions.
    </p>
</div>
""", unsafe_allow_html=True)
