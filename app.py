import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

MODEL_FILE = "model.joblib"
FEATURES_FILE = "feature_columns.json"

st.set_page_config(
    page_title="Vehicle Fuel Efficiency Estimator",
    layout="centered"
)

st.title("🚗 Vehicle Fuel Efficiency Estimator")
st.write("Enter vehicle and driving conditions — model predicts **km per liter**, liters needed for trip, cost, and CO₂.")

model = joblib.load(MODEL_FILE)
with open(FEATURES_FILE, 'r') as f:
    feat_info = json.load(f)

st.header("Vehicle details")
col1, col2 = st.columns(2)

with col1:
    engine_cc = st.number_input(
        "Engine displacement (cc)",
        min_value=50,
        max_value=8000,
        value=1600,
        step=50
    )
    horsepower = st.number_input(
        "Horsepower (HP)",
        min_value=10,
        max_value=1500,
        value=120,
        step=5
    )
    weight_kg = st.number_input(
        "Curb weight (kg)",
        min_value=1,
        max_value=500,
        value=150,
        step=1
    )

with col2:
    fuel_type = st.selectbox(
        "Fuel type",
        ["petrol", "diesel", "hybrid", "electric"],
        index=0
    )
    tyre_psi = st.number_input(
        "Avg tyre pressure (psi)",
        min_value=20.0,
        max_value=45.0,
        value=33.0
    )
    payload_kg = st.number_input(
        "Payload (kg)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )

st.header("Trip & driving conditions")
col3, col4 = st.columns(2)

with col3:
    distance_km = st.number_input(
        "Trip distance (km)",
        min_value=1.0,
        value=50.0
    )
    avg_speed_kmph = st.number_input(
        "Expected avg speed (km/h)",
        min_value=10.0,
        value=50.0
    )

with col4:
    road_type = st.selectbox(
        "Road type",
        ["city", "highway", "hilly"]
    )
    traffic_level = st.selectbox(
        "Traffic level",
        ["low", "medium", "high"]
    )
    ac_on = st.selectbox(
        "AC on during trip?",
        [0, 1],
        index=1
    )

fuel_price = st.number_input(
    "Fuel price (₹ per litre)",
    min_value=1.0,
    value=110.0
)

if st.button("Predict"):
    X = pd.DataFrame([{
        'engine_cc': engine_cc,
        'horsepower': horsepower,
        'weight_kg': weight_kg,
        'road_type': road_type,
        'avg_speed_kmph': avg_speed_kmph,
        'ac_on': ac_on,
        'tyre_psi': tyre_psi,
        'traffic_level': traffic_level,
        'payload_kg': payload_kg
    }])
    
    kmpl = float(model.predict(X)[0])
    liters_needed = distance_km / kmpl if kmpl > 0 else np.nan
    cost = liters_needed * fuel_price
    
    co2_factor = 2.31 if fuel_type == 'petrol' else (2.68 if fuel_type == 'diesel' else 0.0)
    co2_kg = liters_needed * co2_factor if not np.isnan(liters_needed) else np.nan

    st.subheader("Prediction")
    st.metric("Estimated fuel efficiency (km / L)", f"{kmpl:.2f}")
    st.metric("Liters needed for trip", f"{liters_needed:.2f} L")
    st.metric("Estimated fuel cost", f"₹ {cost:.2f}")
    
    if co2_factor > 0:
        st.metric("Estimated CO₂ (kg)", f"{co2_kg:.2f} kg")
    else:
        st.info("CO₂ estimate not available for this fuel type (e.g., electric).")

    st.subheader("Eco tips to improve efficiency")
    tips = []
    
    if avg_speed_kmph > 80 and road_type == 'city':
        tips.append("Avoid high speeds in city — keep steady, lower speed helps efficiency.")
    if ac_on == 1:
        tips.append("Turn off AC when safe; AC increases fuel consumption moderately.")
    if tyre_psi < 30:
        tips.append("Inflate tyres to manufacturer recommended psi (improves fuel economy).")
    if traffic_level == 'high':
        tips.append("Plan trip during off-peak hours to avoid stop-go traffic.")
    if not tips:
        tips.append("Maintain steady speed and correct tyre pressure for best efficiency.")
        
    for t in tips:
        st.write("- " + t)
