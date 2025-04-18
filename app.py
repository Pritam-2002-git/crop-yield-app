import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('C:\\Users\\prita\\Downloads\\crop_yield_model.pkl')
scaler = joblib.load('C:\\Users\\prita\\Downloads\\scaler.pkl')

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; padding: 2rem; border-radius: 10px; }
    .title { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem; color: #228B22; }
    .subtitle { font-size: 1.2rem; text-align: center; margin-bottom: 2rem; color: #444; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🌾 Crop Yield Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter environmental and soil data to predict expected crop yield.</div>', unsafe_allow_html=True)

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=0.1)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, step=0.1)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, step=0.1)

with col2:
    soil_pH = st.number_input("🧪 Soil pH", min_value=0.0, step=0.1)
    nitrogen = st.number_input("🌱 Nitrogen (N)", min_value=0.0, step=0.1)
    phosphorus = st.number_input("🌱 Phosphorus (P)", min_value=0.0, step=0.1)

# --- Predict Button ---
if st.button("📈 Predict Yield"):
    input_data = np.array([rainfall, temperature, humidity, soil_pH, nitrogen, phosphorus]).reshape(1, -1)
    features_scaled = scaler.transform(input_data)
    prediction = model.predict(features_scaled)

    st.success("✅ Prediction Complete!")
    yield_value = prediction[0]
    st.metric("Estimated Yield (Q/acre)", f"{yield_value:.2f}")

    # --- Interpretation ---
    if yield_value >= 35:
        st.success("🌟 Excellent Yield: Great conditions for crop growth!")
    elif 25 <= yield_value < 35:
        st.info("✅ Average Yield: Fairly good, but can be improved.")
    else:
        st.warning("⚠️ Low Yield: May need better inputs or conditions.")

    # --- Save to Session History ---
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    st.session_state['history'].append({
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Humidity': humidity,
        'Soil pH': soil_pH,
        'Nitrogen': nitrogen,
        'Phosphorus': phosphorus,
        'Predicted Yield': yield_value
    })

# --- Show Prediction History ---
if 'history' in st.session_state and len(st.session_state['history']) > 0:
    st.markdown("### 🧾 Prediction History")
    df_history = pd.DataFrame(st.session_state['history'])
    st.dataframe(df_history, use_container_width=True)

    # --- Download CSV Button ---
    csv = df_history.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV Report", data=csv, file_name='prediction_history.csv', mime='text/csv')

# --- Feature Importance Plot ---
try:
    importance = model.feature_importances_
    feature_names = ['Rainfall', 'Temperature', 'Humidity', 'Soil pH', 'Nitrogen', 'Phosphorus']
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

    st.markdown("### 📊 Feature Importance")
    st.bar_chart(importance_df.set_index('Feature'))
except Exception as e:
    st.info("ℹ️ Feature importance is not available for this model.")

st.markdown('</div>', unsafe_allow_html=True)

