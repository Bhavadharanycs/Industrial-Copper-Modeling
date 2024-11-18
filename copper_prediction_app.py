import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import pickle

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("Copper_Set.csv")

# Data cleaning and preprocessing
def clean_data(copper_data):
    # Convert invalid 'material_ref' values starting with '00000' to NaN
    copper_data['material_ref'] = copper_data['material_ref'].replace(
        to_replace=r'^00000.*', value=None, regex=True
    )
    # Convert 'quantity tons' to numeric, forcing errors to NaN for non-numeric values
    copper_data['quantity tons'] = pd.to_numeric(copper_data['quantity tons'], errors='coerce')
    
    # Fill missing values with median for numerical and mode for categorical
    numerical_columns = ['item_date', 'quantity tons', 'customer', 'country', 'application', 
                         'thickness', 'width', 'delivery date', 'selling_price']
    for col in numerical_columns:
        copper_data[col].fillna(copper_data[col].median(), inplace=True)
    
    categorical_columns = ['status', 'item type', 'material_ref']
    for col in categorical_columns:
        copper_data[col].fillna(copper_data[col].mode()[0], inplace=True)
    
    # Treat outliers using Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    outlier_predictions = iso_forest.fit_predict(copper_data[['quantity tons', 'thickness', 'selling_price']])
    copper_data_cleaned = copper_data[outlier_predictions == 1]
    
    # Log transformation for highly skewed columns
    skewed_columns = ['quantity tons', 'thickness', 'selling_price']
    for col in skewed_columns:
        copper_data_cleaned[col] = copper_data_cleaned[col].apply(lambda x: x + 1 if x > 0 else 1)  # Avoid log(0)
        copper_data_cleaned[f'log_{col}'] = np.log(copper_data_cleaned[col])
    
    return copper_data_cleaned

# Encoding categorical variables
def encode_data(copper_data):
    # One-hot encode 'item type'
    encoder = OneHotEncoder(sparse=False, drop='first')
    item_type_encoded = encoder.fit_transform(copper_data[['item type']])
    item_type_df = pd.DataFrame(item_type_encoded, columns=encoder.get_feature_names_out(['item type']))

    # Label encode 'status'
    label_encoder = LabelEncoder()
    copper_data['status_encoded'] = label_encoder.fit_transform(copper_data['status'])

    # Combine data
    copper_data_encoded = pd.concat([copper_data, item_type_df], axis=1).drop(columns=['item type'])
    return copper_data_encoded, label_encoder, encoder

# Model building and training
def train_models(copper_data_encoded):
    # Splitting for Regression
    X_reg = copper_data_encoded.drop(columns=['selling_price', 'status_encoded'])
    y_reg = copper_data_encoded['selling_price']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Splitting for Classification
    X_cls = copper_data_encoded.drop(columns=['status_encoded', 'selling_price'])
    y_cls = copper_data_encoded['status_encoded']
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    # Standard Scaling
    scaler = StandardScaler()
    X_train_reg = scaler.fit_transform(X_train_reg)
    X_test_reg = scaler.transform(X_test_reg)
    X_train_cls = scaler.fit_transform(X_train_cls)
    X_test_cls = scaler.transform(X_test_cls)

    # Regression Model
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)
    regressor_mse = mean_squared_error(y_test_reg, y_pred_reg)

    # Classification Model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_cls, y_train_cls)
    y_pred_cls = classifier.predict(X_test_cls)
    classification_rep = classification_report(y_test_cls, y_pred_cls)

    # Save models
    pickle.dump(regressor, open('regressor_model.pkl', 'wb'))
    pickle.dump(classifier, open('classifier_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(encoder, open('encoder.pkl', 'wb'))
    pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))

    return regressor_mse, classification_rep

# Streamlit app
def run_streamlit():
    st.title("Copper Industry Prediction App")

    # Load and clean data
    copper_data = load_data()
    cleaned_data = clean_data(copper_data)
    encoded_data, label_encoder, encoder = encode_data(cleaned_data)

    # Train models
    regressor_mse, classification_rep = train_models(encoded_data)

    st.write(f"Regression Model MSE: {regressor_mse}")
    st.write(f"Classification Report: \n{classification_rep}")

    # Streamlit inputs and predictions
    task = st.selectbox("Choose Task", ["Regression (Selling Price)", "Classification (Status)"])
    if task == "Regression (Selling Price)":
        # Input fields for regression
        inputs = {col: st.number_input(f"Enter {col}") for col in encoded_data.drop(columns=['selling_price', 'status_encoded']).columns}
        inputs_scaled = scaler.transform([list(inputs.values())])
        prediction = regressor.predict(inputs_scaled)
        st.write(f"Predicted Selling Price: {prediction[0]}")

    elif task == "Classification (Status)":
        # Input fields for classification
        inputs = {col: st.number_input(f"Enter {col}") for col in encoded_data.drop(columns=['status_encoded', 'selling_price']).columns}
        inputs_scaled = scaler.transform([list(inputs.values())])
        prediction = classifier.predict(inputs_scaled)
        status = label_encoder.inverse_transform(prediction)
        st.write(f"Predicted Status: {status[0]}")

if __name__ == "__main__":
    run_streamlit()
