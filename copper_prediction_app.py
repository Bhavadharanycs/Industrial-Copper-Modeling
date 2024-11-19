# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import pickle

# Load dataset
@st.cache
def load_data():
    file_path = "Copper_Set.csv"  # Adjust path as needed
    df = pd.read_csv(file_path)
    df['material_ref'] = df['material_ref'].replace(to_replace=r'^0+$', value=np.nan, regex=True)
    return df

# Preprocessing functions
def preprocess_data(df, task):
    df = df.copy()
    # Drop irrelevant columns
    df.drop(columns=['id', 'product_ref', 'delivery date'], inplace=True)

    # Handle missing values
    for col in ['quantity', 'thickness', 'width']:
        df[col].fillna(df[col].median(), inplace=True)

    # Handle skewness in numeric columns
    skewed_cols = ['quantity', 'thickness', 'width', 'selling_price']
    for col in skewed_cols:
        df[col] = np.log1p(df[col])  # Log transformation

    # Encode categorical variables
    categorical_cols = ['country', 'item type', 'application']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Return features and target based on task
    if task == "Regression":
        X = df.drop(columns=['selling_price', 'status'])
        y = df['selling_price']
    elif task == "Classification":
        df = df[df['status'].isin(['Won', 'Lost'])]
        X = df.drop(columns=['selling_price', 'status'])
        y = df['status'].apply(lambda x: 1 if x == 'Won' else 0)
    return X, y

# Streamlit App
def main():
    st.title("Copper Industry ML Predictions")
    
    task = st.sidebar.selectbox("Choose Task", ["Regression", "Classification"])
    df = load_data()
    X, y = preprocess_data(df, task)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose model
    if task == "Regression":
        model = RandomForestRegressor(random_state=42)
    elif task == "Classification":
        model = RandomForestClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    if task == "Regression":
        y_pred = model.predict(X_test)
        st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    elif task == "Classification":
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

    # Save model
    pickle.dump(model, open(f"{task}_model.pkl", "wb"))

    # Interactive prediction
    st.sidebar.write("Input Features")
    user_input = {col: st.sidebar.text_input(col, "") for col in X.columns}
    user_input_df = pd.DataFrame(user_input, index=[0])

    if st.sidebar.button("Predict"):
        model = pickle.load(open(f"{task}_model.pkl", "rb"))
        prediction = model.predict(user_input_df)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
