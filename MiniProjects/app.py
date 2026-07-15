"""
app.py
Streamlit web app for predicting California house prices
using the trained Random Forest model (house_price_model.pkl).

Run with:
    streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('house_price_model.pkl')

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Predictor")
st.write(
    "This app uses a Random Forest model trained on California housing "
    "data to predict median house value based on the inputs below."
)

col1, col2 = st.columns(2)

with col1:
    median_income = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 5.0, 0.1)
    housing_age = st.slider("Housing Median Age (years)", 1, 52, 20)
    ave_rooms = st.slider("Average Rooms per Household", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.slider("Average Bedrooms per Household", 0.5, 5.0, 1.0, 0.1)

with col2:
    population = st.slider("Block Population", 3, 5000, 1000)
    ave_occup = st.slider("Average Occupancy per Household", 0.5, 10.0, 3.0, 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 36.0, 0.1)
    longitude = st.slider("Longitude", -124.5, -114.0, -119.0, 0.1)

st.markdown("---")

if st.button("Predict Price", type="primary"):
    features = np.array([[
        median_income, housing_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Median House Value: **${prediction:,.2f}**")

st.markdown("---")
st.caption(
    "Model: Random Forest Regressor | Trained on California Housing dataset | "
    "R² ≈ 0.80, RMSE ≈ $50,600"
)
