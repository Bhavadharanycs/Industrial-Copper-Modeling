# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load pre-trained models and preprocessors
@st.cache(allow_output_mutation=True)
def load_models():
    regression_model = pickle.load(open("Regression_model.pkl", "rb"))
    classification_model = pickle.load(open("Classification_model.pkl", "rb"))
    scaler = pickle.load(open("Scaler.pkl", "rb"))
    encoder = pickle.load(open("Encoder.pkl", "rb"))
    return regression_model, classification_model, scaler, encoder

# Streamlit App
def main():
    st.title("Copper Industry Predictions")

    # Sidebar for task selection
    task = st.sidebar.selectbox("Select Task", ["Regression (Selling_Price)", "Classification (Status)"])

    # Feature inputs based on task
    st.sidebar.subheader("Enter Feature Values")
    input_data = {}

    # Common input fields
    input_data["quantity"] = st.sidebar.number_input("Quantity (in tons)", min_value=0.0, value=1.0)
    input_data["thickness"] = st.sidebar.number_input("Thickness (mm)", min_value=0.0, value=1.0)
    input_data["width"] = st.sidebar.number_input("Width (mm)", min_value=0.0, value=100.0)
    input_data["country"] = st.sidebar.selectbox("Country", ["28", "25", "30", "32"])
    input_data["item_type"] = st.sidebar.selectbox("Item Type", ["W", "WI", "S"])
    input_data["application"] = st.sidebar.selectbox("Application", ["10", "41", "28", "59"])

    # Load models and preprocessors
    regression_model, classification_model, scaler, encoder = load_models()

    if st.sidebar.button("Predict"):
        # Prepare data for prediction
        data = pd.DataFrame([input_data])
        # Encode categorical features
        categorical_cols = ["country", "item_type", "application"]
        data[categorical_cols] = encoder.transform(data[categorical_cols])
        # Scale numerical features
        data = scaler.transform(data)

        # Perform predictions
        if "Regression" in task:
            prediction = regression_model.predict(data)
            st.write(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
        elif "Classification" in task:
            prediction = classification_model.predict(data)
            status = "Won" if prediction[0] == 1 else "Lost"
            st.write(f"Predicted Status: {status}")

if __name__ == "__main__":
    main()
