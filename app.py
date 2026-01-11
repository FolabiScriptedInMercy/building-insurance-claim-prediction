import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Building Insurance Claim Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model and artifacts
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

# Header
st.title("🏠 Building Insurance Claim Predictor")
st.markdown("### Predict the likelihood of an insurance claim for a building")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📝 Enter Building Details")

year_of_observation = st.sidebar.selectbox(
    "Year of Observation",
    options=[2012, 2013, 2014, 2015, 2016],
    index=3
)

insured_period = st.sidebar.slider(
    "Insured Period",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.1
)

residential = st.sidebar.selectbox(
    "Residential",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

building_painted = st.sidebar.selectbox(
    "Building Painted",
    options=["V", "N"],
    format_func=lambda x: "Yes (V)" if x == "V" else "No (N)"
)

building_fenced = st.sidebar.selectbox(
    "Building Fenced",
    options=["V", "N"],
    format_func=lambda x: "Yes (V)" if x == "V" else "No (N)"
)

garden = st.sidebar.selectbox(
    "Garden",
    options=["V", "O"],
    format_func=lambda x: "Yes (V)" if x == "V" else "No (O)"
)

settlement = st.sidebar.selectbox(
    "Settlement Type",
    options=["U", "R"],
    format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)"
)

building_dimension = st.sidebar.number_input(
    "Building Dimension (sqm)",
    min_value=1,
    max_value=25000,
    value=1500
)

building_type = st.sidebar.selectbox(
    "Building Type",
    options=[1, 2, 3, 4]
)

date_of_occupancy = st.sidebar.number_input(
    "Date of Occupancy (Year)",
    min_value=1800,
    max_value=2016,
    value=1990
)

number_of_windows = st.sidebar.slider(
    "Number of Windows",
    min_value=1,
    max_value=10,
    value=4
)

geo_code = st.sidebar.text_input(
    "Geo Code",
    value="6088"
)

# Prediction function
def predict_claim(input_data):
    df = input_data.copy()
    
    # Create Age_of_Building
    df['Age_of_Building'] = df['YearOfObservation'] - df['Date_of_Occupancy']
    df['Age_of_Building'] = df['Age_of_Building'].clip(lower=0)
    
    # Target encode Geo_Code
    df['Geo_Code_TargetEnc'] = df['Geo_Code'].astype(str).map(geo_encoding)
    df['Geo_Code_TargetEnc'].fillna(geo_encoding.mean(), inplace=True)
    df = df.drop('Geo_Code', axis=1)
    
    # Label encode categorical columns
    for col in ['Building_Painted', 'Building_Fenced', 'Garden', 'Settlement']:
        df[col] = label_encoders[col].transform(df[col].astype(str))
    
    # Ensure correct column order
    feature_order = [
        'YearOfObservation', 'Insured_Period', 'Residential', 
        'Building_Painted', 'Building_Fenced', 'Garden', 'Settlement',
        'Building Dimension', 'Building_Type', 'Date_of_Occupancy',
        'NumberOfWindows', 'Age_of_Building', 'Geo_Code_TargetEnc'
    ]
    df = df[feature_order]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    probability = model.predict_proba(df_scaled)[:, 1][0]
    prediction = 1 if probability >= threshold else 0
    
    return probability, prediction

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Summary")
    
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
    
    st.dataframe(input_data.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("🎯 Prediction Result")
    
    if st.button("🔮 Predict Claim", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            probability, prediction = predict_claim(input_data)
        
        # Probability meter
        st.metric(label="Claim Probability", value=f"{probability:.1%}")
        st.progress(probability)
        
        # Prediction result
        if prediction == 1:
            st.error("⚠️ **CLAIM PREDICTED**")
            st.markdown("The model predicts this building is **likely to file a claim**.")
        else:
            st.success("✅ **NO CLAIM PREDICTED**")
            st.markdown("The model predicts this building is **unlikely to file a claim**.")
        
        # Risk level
        st.markdown("---")
        st.subheader("📈 Risk Level")
        
        if probability >= 0.6:
            st.error("🔴 **HIGH RISK**")
        elif probability >= 0.4:
            st.warning("🟡 **MEDIUM RISK**")
        else:
            st.success("🟢 **LOW RISK**")
        
        st.info(f"**Threshold Used:** {threshold}")

# Footer
st.markdown("---")
st.markdown("### ℹ️ Model Performance")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("F1-Score", "0.5355")
with col2:
    st.metric("Recall", "70.3%")
with col3:
    st.metric("ROC-AUC", "0.8025")

st.markdown("**Model:** LightGBM | **Created by:** FolabiScriptedInMercy")
