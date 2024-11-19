import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
from sklearn.ensemble import IsolationForest
import pickle
import streamlit as st

# Load Dataset
file_path = 'Copper_Set.csv'  # Replace with uploaded file path
data = pd.read_csv(file_path)

# Data Preprocessing
# Treat Material_Reference rubbish values as null
data['Material_Reference'] = data['Material_Reference'].replace(r'^00000.*', np.nan, regex=True)

# Handle Missing Values
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Remove INDEX column if exists
if 'INDEX' in data.columns:
    data.drop(columns=['INDEX'], inplace=True)

# Treat Skewness in Target Variable (Selling_Price) using log transformation
data['Selling_Price'] = np.log1p(data['Selling_Price'])

# Encode Categorical Variables
categorical_cols = data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Treat Outliers using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(data)
data = data[outliers == 1]

# Feature and Target Separation
X_reg = data.drop(columns=['Selling_Price', 'Status'])
y_reg = data['Selling_Price']
X_cls = data.drop(columns=['Status', 'Selling_Price'])
y_cls = data['Status']

# Train-Test Split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_reg_train_scaled = scaler.fit_transform(X_reg_train)
X_reg_test_scaled = scaler.transform(X_reg_test)

# Model Training
regressor = RandomForestRegressor(random_state=42)
classifier = RandomForestClassifier(random_state=42)
regressor.fit(X_reg_train_scaled, y_reg_train)
classifier.fit(X_cls_train, y_cls_train)

# Save Models
pickle.dump(regressor, open('regressor.pkl', 'wb'))
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
    st.write(f"Predicted Status: {'WON' if pred[0] == 1 else 'LOST'}")
