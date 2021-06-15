import streamlit as st
import pickle
import numpy as np


def predict(*args):
    features = [item for item in args]

    model = pickle.load(open("model.pkl", 'rb'))
    scaler = pickle.load(open("scaler/scaler.pkl", 'rb'))
    encoder = pickle.load(open("labelencoder/labelencoder.pkl", 'rb'))
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)

    return encoder.inverse_transform(prediction)[0]


st.title('What is the best crop to cultivate?')

with st.form("crops"):
    N = st.slider("N ratio", help="Ratio of Nitrogen content in soil", min_value=0, max_value=100)
    P = st.slider("P ratio", help="Ratio of Phosphorous content in soil", min_value=0, max_value=100)
    K = st.slider("K ratio", help="Ratio of Potassium content in soil", min_value=0, max_value=100)
    humidity = st.slider("Humidity %", help="Relative humidity in %", min_value=0.0, max_value=100.0)
    ph = st.slider("ph", help="ph value of the soil", min_value=0.0, max_value=14.0)
    temperature = st.number_input("Temperature (Celsius)", help="Temperature in degree Celsius", min_value=0.0)
    rainfall = st.number_input("Rainfall mm", help="rainfall in mm", min_value=0.0)

    submitted = st.form_submit_button("Submit")
    if submitted:
        predictions = predict(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"Suggested crop to cultivate: {predictions.capitalize()}")
