import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import pickle
import streamlit as st

# Load Dataset
file_path = 'Copper_Set.csv'  # Replace with the uploaded file path
data = pd.read_csv(file_path)

# Data Preprocessing
# Handle mixed data types by coercing object columns to numeric where applicable
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # Use mean imputation
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Remove the 'INDEX' column if it exists
if 'INDEX' in data.columns:
    data.drop(columns=['INDEX'], inplace=True)

# Handle Skewness in 'Selling_Price' using log transformation
if 'Selling_Price' in data.columns:
    data['Selling_Price'] = np.log1p(data['Selling_Price'])

# Treat outliers using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(data)
data = data[outliers == 1]

# Separate Regression and Classification Tasks
if 'Selling_Price' in data.columns:
    X_reg = data.drop(columns=['Selling_Price', 'Status'], errors='ignore')
    y_reg = data['Selling_Price']

if 'Status' in data.columns:
    X_cls = data.drop(columns=['Status', 'Selling_Price'], errors='ignore')
    y_cls = data['Status']

# Encode categorical variables
encoder = LabelEncoder()
if 'Status' in locals():
    y_cls = encoder.fit_transform(y_cls)

# Train-Test Split
if 'Selling_Price' in locals():
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

if 'Status' in locals():
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
if 'Selling_Price' in locals():
    X_reg_train_scaled = scaler.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler.transform(X_reg_test)

# Train Models
if 'Selling_Price' in locals():
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_reg_train_scaled, y_reg_train)

if 'Status' in locals():
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_cls_train, y_cls_train)

# Save Models and Preprocessing Objects
if 'Selling_Price' in locals():
    pickle.dump(regressor, open('regressor.pkl', 'wb'))
if 'Status' in locals():
    pickle.dump(classifier, open('classifier.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Streamlit App
st.title("Copper Industry ML Application")
task = st.selectbox("Select Task", ["Regression (Selling Price)", "Classification (Status)"])

if task == "Regression (Selling Price)":
    st.header("Predict Selling Price")
    input_data = []
    for col in X_reg.columns:
        value = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(value)
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    regressor = pickle.load(open('regressor.pkl', 'rb'))
    pred = regressor.predict(input_data_scaled)
    st.write(f"Predicted Selling Price: {np.expm1(pred[0]):.2f} (Original Scale)")

elif task == "Classification (Status)":
    st.header("Predict Status (Won/Lost)")
    input_data = []
    for col in X_cls.columns:
        value = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(value)
    input_data = np.array(input_data).reshape(1, -1)
    classifier = pickle.load(open('classifier.pkl', 'rb'))
    pred = classifier.predict(input_data)
    status = 'WON' if pred[0] == 1 else 'LOST'
    st.write(f"Predicted Status: {status}")
