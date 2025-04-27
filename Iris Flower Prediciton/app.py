import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_species(model, scaler, features):
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return prediction


# Load model and scaler
model, scaler = load_model()

# Streamlit UI
st.title("Iris Flower Species Prediction")
st.write("Enter feature values to predict the species:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict_species(model, scaler, features)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.write(f"Predicted Species: {species_map[prediction]}")
