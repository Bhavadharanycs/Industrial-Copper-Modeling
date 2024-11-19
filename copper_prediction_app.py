# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle

# Function to preprocess data
def preprocess_data(df, task):
    # Handle missing values in 'material_ref'
    df['material_ref'] = df['material_ref'].replace(to_replace=r'^0+$', value=np.nan, regex=True)
    
    # Drop unnecessary columns
    df.drop(columns=["id", "product_ref", "delivery date", "material_ref"], inplace=True)

    # Handle missing values
    for col in ["quantity", "thickness", "width"]:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode target variable for classification
    if task == "Classification":
        df = df[df['status'].isin(["Won", "Lost"])]
        df['status'] = df['status'].apply(lambda x: 1 if x == "Won" else 0)
    
    return df

# Function to build models
def build_and_save_models(df):
    # Separate features and targets
    X = df.drop(columns=["selling_price", "status"])
    y_reg = df["selling_price"]
    y_cls = df["status"]

    # Define categorical and numerical columns
    categorical_cols = ["country", "item type", "application"]
    numerical_cols = ["quantity", "thickness", "width"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Regression model
    reg_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))])
    reg_model.fit(X, y_reg)
    pickle.dump(reg_model, open("Regression_model.pkl", "wb"))

    # Classification model
    cls_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(random_state=42))])
    cls_model.fit(X, y_cls)
    pickle.dump(cls_model, open("Classification_model.pkl", "wb"))

# Streamlit App
def main():
    st.title("Copper Industry ML Predictions")

    # Sidebar for task selection
    task = st.sidebar.selectbox("Select Task", ["Regression (Selling_Price)", "Classification (Status)"])
    
    # Upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df, task)
        build_and_save_models(df)
        st.success("Models trained and saved!")

    # Load models
    reg_model = pickle.load(open("Regression_model.pkl", "rb"))
    cls_model = pickle.load(open("Classification_model.pkl", "rb"))

    # Sidebar inputs
    st.sidebar.subheader("Enter Feature Values")
    input_data = {
        "quantity": st.sidebar.number_input("Quantity (in tons)", min_value=0.0, value=1.0),
        "thickness": st.sidebar.number_input("Thickness (mm)", min_value=0.0, value=1.0),
        "width": st.sidebar.number_input("Width (mm)", min_value=0.0, value=100.0),
        "country": st.sidebar.selectbox("Country", ["28", "25", "30", "32"]),
        "item type": st.sidebar.selectbox("Item Type", ["W", "WI", "S"]),
        "application": st.sidebar.selectbox("Application", ["10", "41", "28", "59"]),
    }

    # Interactive prediction
    if st.sidebar.button("Predict"):
        data = pd.DataFrame([input_data])

        # Select model based on task
        model = reg_model if "Regression" in task else cls_model
        prediction = model.predict(data)

        if "Regression" in task:
            st.write(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
        elif "Classification" in task:
            status = "Won" if prediction[0] == 1 else "Lost"
            st.write(f"Predicted Status: {status}")

if __name__ == "__main__":
    main()
