import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from demand import ARIMAWrap

def app(df):
    # Load the pre-trained data cleaning pipeline and ARIMA model pipeline
    data_cleaning_pipeline = joblib.load('data_cleaning_pipeline.pkl')
    arima_pipeline = joblib.load('demand_pipeline.joblib')

    # Apply the data cleaning pipeline to the uploaded dataset
    df_cleaned = data_cleaning_pipeline.transform(df)

    # Convert 'timestamp' to datetime and extract year, month, and day
    df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], errors='coerce')
    df_cleaned['year'] = df_cleaned['timestamp'].dt.year
    df_cleaned['month'] = df_cleaned['timestamp'].dt.month
    df_cleaned['day'] = df_cleaned['timestamp'].dt.day

    # Ensure there are no NaN values in the timestamp columns
    df_cleaned = df_cleaned.dropna(subset=['timestamp'])

    # Group by year, month, day, product category, and product ID, aggregating the quantity
    df_grouped = df_cleaned.groupby(['timestamp', 'year', 'month', 'day', 'productcategory', 'productid']).agg({'quantity': 'sum'}).reset_index()

    st.write("## Demand Forecasting")

    # Streamlit interface for selecting product category
    product_categories = df_grouped['productcategory'].unique().tolist()
    selected_category = st.selectbox("Product Category", product_categories)

    # Filter data based on the selected category
    filtered_data_by_category = df_grouped[df_grouped['productcategory'] == selected_category]

    if filtered_data_by_category.empty:
        st.error("No data available for the selected product category.")
    else:
        # Streamlit interface for selecting product within the selected category
        product_ids = filtered_data_by_category['productid'].unique().tolist()
        selected_product = st.selectbox("Select Product", product_ids)

        if selected_product:
            # Filter data based on the selected product
            filtered_data = filtered_data_by_category[filtered_data_by_category['productid'] == selected_product]

            if filtered_data.empty:
                st.error("No data available for the selected product.")
            else:
                # Prepare features for prediction using the ARIMA model
                X_filtered = filtered_data[['year', 'month', 'day']]

                # Create a dataframe for the forecasted years
                last_date = filtered_data['timestamp'].max()
                forecast_dates = pd.date_range(start=last_date, periods=365, freq='D')
                forecast_years = pd.DataFrame({
                    'year': forecast_dates.year,
                    'month': forecast_dates.month,
                    'day': forecast_dates.day,
                    'productcategory': [selected_category] * len(forecast_dates),
                    'productid': [selected_product] * len(forecast_dates)  # Include productid for consistency
                })

                # Predict future demand
                future_pred = arima_pipeline.predict(forecast_years)

                # Combine historical and forecasted data
                forecasted_data = pd.DataFrame({
                    'timestamp': list(filtered_data['timestamp']) + list(forecast_dates),
                    'quantity': list(filtered_data['quantity']) + list(future_pred),
                    'type': ['Historical'] * len(filtered_data) + ['Forecast'] * len(forecast_dates)
                })

                st.write("### Forecasted Values")
                st.write(forecasted_data)

                # Plot historical and forecasted values
                plt.figure(figsize=(10, 5))
                plt.plot(filtered_data['timestamp'], filtered_data['quantity'], label='Historical', color='red')
                plt.plot(forecast_dates, future_pred, label='Forecast', color='blue')
                plt.xlabel('Date')
                plt.ylabel('Quantity')
                plt.title(f'Demand Forecasting for Product ID: {selected_product} in Category: {selected_category}')
                plt.legend()
                st.pyplot(plt)
