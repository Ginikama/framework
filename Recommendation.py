import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Define preprocessing pipeline
def preprocess_data(df):
    numerical_columns = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'HourSpendOnApp',
                         'NumberOfDeviceRegistered', 'OrderAmountHikeFromlastYear', 'OrderCount', 'DaySinceLastOrder',
                         'CashbackAmount', 'Sales', 'Discount', 'Profit', 'Total Spend', 'Items Purchased', 'Average Rating']

    categorical_columns = ['Product Category', 'Gender', 'Country', 'PreferredLoginDevice', 'PreferedOrderCat', 
                           'Product ID', 'Sub-Category', 'Product Name', 'Membership Type', 'Satisfaction Level']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )

    X = preprocessor.fit_transform(df)
    return X, preprocessor

# Function to make recommendations using Linear Regression model
def make_recommendations(user_data, lr_model, preprocessor, df):
    user_data_processed = preprocessor.transform(user_data)
    predicted_purchase_amounts = lr_model.predict(user_data_processed)
    user_data['Predicted Purchase Amount'] = predicted_purchase_amounts
    recommended_products = user_data.sort_values(by='Predicted Purchase Amount', ascending=False)
    
    # Extract product category of the user
    user_category = user_data['PreferedOrderCat'].values[0]
    
    # Filter the DataFrame to include only products in the preferred order category
    products_in_category = df[df['PreferedOrderCat'] == user_category]['Product Name'].unique()
    
    return recommended_products[['Product ID', 'Predicted Purchase Amount']], products_in_category

# Streamlit app
def app(df):
    st.title("Personalized Product Recommendations")
    unique_customer_ids = df['Customer ID'].unique()
    selected_customer_id = st.selectbox("Select Customer ID:", options=unique_customer_ids)
    selected_customer_data = df[df['Customer ID'] == selected_customer_id]
    st.subheader("Selected Customer Data:")
    st.write(selected_customer_data)
    
    # Preprocess data
    X, preprocessor = preprocess_data(df)
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['PreferedOrderCat'])
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X, y_encoded)
    
    if st.button("Generate Recommendations"):
        # Filter the DataFrame to include only the relevant features for the user
        user_data = df[df['Customer ID'] == selected_customer_id].copy()
        
        # Get the preferred order category for the user
        user_category = user_data['PreferedOrderCat'].values[0]
        
        # Filter the DataFrame to include only products in the preferred order category
        user_data = df[df['PreferedOrderCat'] == user_category].copy()
        
        # Drop the 'Customer ID' column as it's not needed for prediction
        user_data = user_data.drop('Customer ID', axis=1)
        
        # Make recommendations
        recommended_products, products_in_category = make_recommendations(user_data, lr_model, preprocessor, df)
        
        # Display recommended products with their predicted purchase amounts
        st.subheader("Recommended Products:")
        st.write(products_in_category)
        
       
        

