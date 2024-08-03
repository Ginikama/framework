import streamlit as st
import pandas as pd
import joblib
from transformerSegment import FeatureSelector, EncodeCategorical, StandardizeFeatures
import matplotlib.pyplot as plt

# Load pipelines
cleaning_pipeline = joblib.load('data_cleaning_pipeline.pkl')
segment_pipeline = joblib.load('segment_pipeline.pkl')

# Define cluster names
cluster_names = {
    0: 'Occasional Discount Shoppers',
    1: 'Young Active Shoppers',
    2: 'Satisfied Regular Buyers',
    3: 'Average Value Customers',
    4: 'High-Spending Infrequent Buyers',
    5: 'Discount-Driven Shoppers',
    6: 'Premium Loyal Customers'
}

def app(df):
    

    if df is not None:
                
        try:
            df_cleaned = cleaning_pipeline.transform(df)
            # Apply the segmentation pipeline
            st.subheader("Segmented Data")
            df_segmented = segment_pipeline.fit_predict(df_cleaned)
            df_cleaned['Cluster'] = df_segmented
            df_cleaned['ClusterName'] = df_cleaned['Cluster'].map(cluster_names)
            st.write(df_cleaned.head())


            # Show pie chart for cluster distribution
            st.subheader("Cluster Distribution")
            cluster_counts = df_cleaned['ClusterName'].value_counts()

            fig, ax = plt.subplots()
            ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig)

            # Let user select a variable for segmentation
            st.subheader("Select Variable for Segmentation Analysis")
            variable = st.selectbox('Choose a variable:', df_cleaned.columns)

            # Display customer details for each segment
            st.subheader("Customer Details by Segment")
            selected_cluster_name = st.selectbox('Choose a cluster name:', df_cleaned['ClusterName'].unique())
            selected_cluster = [key for key, value in cluster_names.items() if value == selected_cluster_name][0]

            # Filter the DataFrame for the selected cluster
            df_cluster = df_cleaned[df_cleaned['Cluster'] == selected_cluster]
            st.write(df_cluster)

            # Show bar chart for the selected variable
            st.subheader(f"Distribution of {variable} in Cluster {selected_cluster_name}")
            st.bar_chart(df_cluster[variable].value_counts())

           

        except Exception as e:
            st.error(f"An error occurred: {e}")
