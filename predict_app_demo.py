import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv('mat3d_raw_data.csv')
    df = df[['Ad LTV D7', 'IAP LTV D7', 'Play Time D1', 'IAP LTV D1', 'Ad LTV D1', 'Attributed eCPI',
             'Banner RPM', 'Inter. RPM', 'Reward. RPM', 'Banner IPDAU', 'Inter. IPDAU', 'Reward. IPDAU',
             'Sessions', 'Ad LTV D360']]
    df = df.dropna()
    return df

df = load_data()
X, y = df.iloc[:, :-1], df.iloc[:, -1]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply RobustScaler
scaler_X = RobustScaler()
scaler_y = RobustScaler()
train_X_scaled = scaler_X.fit_transform(train_X)
test_X_scaled = scaler_X.transform(test_X)
train_y_scaled = scaler_y.fit_transform(train_y.values.reshape(-1, 1)).flatten()
test_y_scaled = scaler_y.transform(test_y.values.reshape(-1, 1)).flatten()

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X_scaled, train_y_scaled)

# GUI using Streamlit
st.title("Predict LTV D360")
st.subheader("Upload your data for prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data:")
    st.dataframe(user_data)
    
    # Ensure columns match the trained model
    required_features = X.columns.tolist()
    if set(required_features).issubset(user_data.columns):
        user_data = user_data[required_features]
        user_data = user_data.dropna()
        
        # Scale input data
        user_data_scaled = scaler_X.transform(user_data)
        predictions_scaled = model.predict(user_data_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
        
        st.write("Predictions for LTV D360:")
        user_data["Predicted LTV D360"] = predictions
        st.dataframe(user_data)
    else:
        st.error("Uploaded data does not have the required features. Please check your file.")
