import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Preprocess data
def preprocess_data(df, selected_features):
    # Define the categorical and numerical columns
    categorical_columns = [col for col in selected_features if df[col].dtype == 'object']
    numerical_columns = [col for col in selected_features if df[col].dtype != 'object']

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Applying the transformations
    preprocessed_df = preprocessor.fit_transform(df[selected_features])

    return preprocessed_df

# Perform clustering
def perform_clustering(df, n_clusters):
    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = clustering.fit_predict(df)
    return clusters, clustering

def visualize_results(df, clusters, selected_var):
    st.write("Customer Segmentation Results:")
    graph_type = st.selectbox('Select graph type', ['Scatter Plot', 'Histogram'])

    if graph_type == 'Scatter Plot':
        visualize_scatter_plot(df, clusters, selected_var)
    elif graph_type == 'Histogram':
        visualize_histogram(df, clusters, selected_var)

def visualize_scatter_plot(df, clusters, selected_var):
    plt.figure(figsize=(10, 6))
    col_index = df.columns.get_loc(selected_var)
    plt.scatter(df[selected_var], clusters, c=clusters, cmap='viridis')
    plt.xlabel(selected_var)
    plt.ylabel('Cluster')
    plt.title('Customer Segmentation - Scatter Plot')
    st.pyplot(plt)

def visualize_histogram(df, clusters, selected_var):
    plt.figure(figsize=(15, 8))
    for i in range(max(clusters) + 1):
        plt.hist(df.loc[clusters == i, selected_var], bins=30, alpha=0.5, label=f'Cluster {i}')
    plt.xlabel(selected_var)
    plt.ylabel('Frequency')
    plt.title('Customer Segmentation - Histogram')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

def app(df):
    st.title("Customer Segmentation App")
    
    # User selects features for clustering
    selected_features = st.multiselect(
        'Select features for clustering', 
        df.columns.tolist(), 
        default=df.columns.tolist()
    )
    
    # Preprocess data
    preprocessed_df = preprocess_data(df, selected_features)
    
    # User selects number of clusters
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    
    # Perform clustering
    clusters, model = perform_clustering(preprocessed_df, n_clusters)
    
    # User selects variable for visualization
    selected_var = st.selectbox('Select variable for segmentation', df.columns)
    
    # Visualize results
    visualize_results(df, clusters, selected_var)

