import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Function to add 'is_fraud' column based on LOF
def add_is_fraud_column(df):
    features = df[['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age',
    'Churn', 'UnitPrice', 'WarehouseToHome', 'HourSpendOnApp', 
    'NumberOfDeviceRegistered', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
    'payment', 'rating', 'payment_installments', 'Sales', 'Discount', 
    'Profit', 'Total Spend', 'Items Purchased', 'Average Rating', 
    'Days Since Last Purchase']]
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(features)
    df['is_fraud'] = anomaly_scores
    df['is_fraud'] = df['is_fraud'].apply(lambda x: 1 if x == -1 else 0)
    return df

# Function to preprocess data
def preprocess_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Define the Streamlit app
def app(df):
    st.title("Fraud Detection")

    # Add is_fraud column to the dataframe
    df = add_is_fraud_column(df)

    # Preprocess the dataframe
    df = preprocess_data(df)

    # Split the data into features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Gradient Boosting model with best parameters
    best_params = {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 50, 'subsample': 0.8}
    model = GradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train)

    # Select customer ID
    customer_id = st.selectbox("Select Customer ID", df['Customer ID'].unique())

    # Filter dataframe based on selected customer ID
    selected_customer_data = df[df['Customer ID'] == customer_id]

    # Display all rows related to the selected customer ID
    st.subheader("All Rows Related to Selected Customer ID")
    st.write(selected_customer_data)

    # Allow only one row to be selected at a time
    selected_index = st.radio("Select Row", selected_customer_data.index)

    # Retrieve selected row
    selected_row = selected_customer_data.loc[selected_index]

    # Display the selected row
    st.subheader("Selected Row")
    st.write(selected_row)

    # Preprocess the selected row for prediction
    selected_row_data = selected_row.drop('is_fraud').to_frame().T
    selected_row_data = preprocess_data(selected_row_data)
    preprocessed_row = scaler.transform(selected_row_data)

    # Predict fraud for the selected row
    predicted_fraud = model.predict(preprocessed_row)

    # Display prediction result
    st.subheader("Prediction Result")
    if predicted_fraud[0] == 1:
        st.write("Predicted Fraud: ", predicted_fraud[0], " (Fraudulent)")
    else:
        st.write("Predicted Fraud: ", predicted_fraud[0], " (Not Fraudulent)")


