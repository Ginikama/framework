import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def app(df):
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    selected_columns = ['timestamp', 'Quantity', 'Product Category', 'Product Name', 'Total Purchase Amount']
    df_forecasting = df[selected_columns]

    # Streamlit interface
    st.title("Demand Forecasting")

    # Input variables
    st.subheader("Input Variables")
    
    # Product Category input as a dropdown menu
    product_categories = df['Product Category'].unique().tolist()
    selected_category = st.selectbox("Product Category", product_categories)
    
    # Filter product names based on selected category
    filtered_product_names = df[df['Product Category'] == selected_category]
    # selected_product_name = st.selectbox("Product Name", filtered_product_names)
    
    if st.button("Forecast"):
        # Filter data based on selected category and product name
        filtered_data = df_forecasting[(df_forecasting['Product Category'] == selected_category) ]
        
        # Print data info for debugging
        st.write("Filtered data shape:", filtered_data.shape)
        st.write("Filtered data sample:")
        st.write(filtered_data.head())

        if filtered_data.shape[0] < 2:  # Check if there are enough samples
            st.write("Not enough data for forecasting. Please select a different product or category.")
            return
        
        # Extract year and month from 'timestamp' and assign it to the DataFrame using .loc
        filtered_data['Year'] = filtered_data['timestamp'].dt.to_period('Y')

        # Group by 'Year-Month' and calculate total quantity purchased
        demand_over_time = filtered_data.groupby('Year')['Quantity'].sum().reset_index()

        # Convert 'Quantity' column to numeric type, handling non-numeric values as NaN
        demand_over_time['Quantity'] = pd.to_numeric(demand_over_time['Quantity'], errors='coerce')

        # Plot demand over time using a bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(demand_over_time['Year'].astype(str), demand_over_time['Quantity'])
        plt.title('Demand Trends Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Quantity Purchased')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

        # Ensure the 'Total Purchase Amount' column is numeric
        df_forecasting['Total Purchase Amount'] = pd.to_numeric(df_forecasting['Total Purchase Amount'], errors='coerce')

        # Select features (X) and target variable (y)
        X = filtered_data.drop(columns=['Total Purchase Amount', 'timestamp', 'Year'])
        y = filtered_data['Total Purchase Amount']

        # Define preprocessing for numeric features
        numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Preprocess the data
        X_preprocessed = preprocessor.fit_transform(X)

        # Split the data into training and testing sets
        if len(y) > 1:  # Ensure there is enough data to split
            X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
        else:
            st.write("Not enough data for splitting into training and testing sets.")
            return

        # Fit Holt-Winters Exponential Smoothing model
        hw_model = ExponentialSmoothing(y_train, seasonal_periods=12, trend=None, seasonal='mul').fit()

        # Use the model for final forecasting
        forecast_periods = 72
        forecast = hw_model.forecast(steps=forecast_periods)

        # Plot the results
        plt.figure(figsize=(12, 6))
        actual_dates = filtered_data.iloc[:len(y_train)]['timestamp']
        plt.plot(actual_dates, y_train, label='Actual', color='blue')

        # Generate index for forecasted values
        last_date = filtered_data['timestamp'].max()
        forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_periods, freq='M')
        plt.plot(forecast_index, forecast, label='Forecast', color='red')

        # Set plot title and labels
        plt.title(f'Demand Forecasting for {selected_category}')
        plt.xlabel('Time')
        plt.ylabel('Total Purchase Amount')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

        # Display the forecast results
        st.subheader(f"Forecast for the next {forecast_periods} Months:")
        st.write(forecast)


