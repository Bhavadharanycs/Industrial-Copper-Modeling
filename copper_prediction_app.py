import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

def preprocess_data(df, task):
    # Display dataset columns for debugging
    st.write("Dataset Columns:", df.columns.tolist())
    
    # Handle missing or irrelevant columns dynamically
    if "material_ref" in df.columns:
        df['material_ref'] = df['material_ref'].replace(to_replace=r'^0+$', value=np.nan, regex=True)

    # Drop unnecessary columns if they exist
    drop_columns = ["id", "product_ref", "delivery date", "material_ref"]
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    # Handle missing values for numeric columns
    numeric_cols = ["quantity", "thickness", "width"]
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Filter and encode target variable for classification
    if task == "Classification" and "status" in df.columns:
        df = df[df['status'].isin(["Won", "Lost"])]
        df['status'] = df['status'].apply(lambda x: 1 if x == "Won" else 0)

    return df
    
def load_or_train_models(df):
    # Check if models exist, load them if available
    if os.path.exists("Regression_model.pkl") and os.path.exists("Classification_model.pkl"):
        reg_model = pickle.load(open("Regression_model.pkl", "rb"))
        cls_model = pickle.load(open("Classification_model.pkl", "rb"))
    else:
        # Separate features and targets
        X = df.drop(columns=["selling_price", "status"], errors="ignore")
        y_reg = df["selling_price"] if "selling_price" in df.columns else None
        y_cls = df["status"] if "status" in df.columns else None

        # Handle missing target values and ensure alignment
        if y_reg is not None:
            valid_idx = ~y_reg.isna()
            X = X.loc[valid_idx].reset_index(drop=True)
            y_reg = y_reg.loc[valid_idx].reset_index(drop=True)
            st.write(f"Dropped {len(valid_idx) - valid_idx.sum()} rows with missing 'selling_price'.")

        if y_cls is not None:
            valid_idx = ~y_cls.isna()
            X = X.loc[valid_idx].reset_index(drop=True)
            y_cls = y_cls.loc[valid_idx].reset_index(drop=True)
            st.write(f"Dropped {len(valid_idx) - valid_idx.sum()} rows with missing 'status'.")

        # Define categorical and numerical columns
        categorical_cols = ["country", "item type", "application"]
        numerical_cols = ["quantity", "thickness", "width"]

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [col for col in numerical_cols if col in X.columns]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), [col for col in categorical_cols if col in X.columns]),
            ]
        )

        # Train regression model
        reg_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))])
        if y_reg is not None and len(y_reg) > 0:
            reg_model.fit(X, y_reg)
            pickle.dump(reg_model, open("Regression_model.pkl", "wb"))

        # Train classification model
        cls_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(random_state=42))])
        if y_cls is not None and len(y_cls) > 0:
            cls_model.fit(X, y_cls)
            pickle.dump(cls_model, open("Classification_model.pkl", "wb"))
    
    return reg_model, cls_model
    
# Streamlit App
def main():
    st.title("Copper Industry ML Prediction App")

    # Sidebar task selection
    task = st.sidebar.selectbox("Select Task", ["Regression (Selling_Price)", "Classification (Status)"])
    
    # Upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        # Load and preprocess the dataset
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df, task)

        # Load or train models
        reg_model, cls_model = load_or_train_models(df)
        st.success("Models loaded or trained successfully!")

        # Sidebar inputs for prediction
        st.sidebar.subheader("Enter Feature Values")
        input_data = {
            "quantity": st.sidebar.number_input("Quantity (in tons)", min_value=0.0, value=1.0),
            "thickness": st.sidebar.number_input("Thickness (mm)", min_value=0.0, value=1.0),
            "width": st.sidebar.number_input("Width (mm)", min_value=0.0, value=100.0),
            "country": st.sidebar.selectbox("Country", ["28", "25", "30", "32"]),
            "item type": st.sidebar.selectbox("Item Type", ["W", "WI", "S"]),
            "application": st.sidebar.selectbox("Application", ["10", "41", "28", "59"]),
        }

        # Prepare input data
        input_df = pd.DataFrame([input_data])

        # Prediction
        if st.sidebar.button("Predict"):
            model = reg_model if "Regression" in task else cls_model
            prediction = model.predict(input_df)

            if "Regression" in task:
                st.write(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
            elif "Classification" in task:
                status = "Won" if prediction[0] == 1 else "Lost"
                st.write(f"Predicted Status: {status}")

if __name__ == "__main__":
    main()
