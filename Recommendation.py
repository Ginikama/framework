import streamlit as st
import pandas as pd
import joblib

# Load the saved model
pipeline_best_model = joblib.load('recommender.joblib')

# Load the dataset to get product information
data = pd.read_csv('cleaned_data.csv')

# Define the columns to be used for recommendation input
input_columns = [
    'customerid', 'gender', 'age', 'city', 'membershiptype',
    'itemspurchased', 'dayssincelastpurchase', 'hourspendonapp',
    'productcategory', 'productprice', 'quantity', 'ordercount',
    'couponused', 'cashbackamount'
]

def recommend_products(pipeline, input_df):
    # Preprocess input data
    X_input_transformed = pipeline.named_steps['preprocessor'].transform(input_df)
    if 'svd' in pipeline.named_steps:
        X_input_transformed = pipeline.named_steps['svd'].transform(X_input_transformed)
    
    # Get recommendations
    distances, indices = pipeline.named_steps['model'].kneighbors(X_input_transformed)
    return distances, indices

def app():
       # User inputs
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=12, max_value=100, value=25)
    city = st.text_input("City")
    membershiptype = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold"])
   
    dayssincelastpurchase = st.slider("Days Since Last Purchase", min_value=0, value=0)
    couponused = st.number_input("Coupon Used", min_value=0.0, value=0.0)
    productcategory = st.text_input("Product Category")
    productprice = st.number_input("Product Price", min_value=0.0, value=0.0)
    quantity = st.number_input("Quantity", min_value=0, value=0)
    hourspendonapp = st.slider("Hours Spent on App", min_value=0, value=0 )
    ordercount = st.number_input("Order Count", min_value=0, value=0)
    itemspurchased = st.slider("Items Purchased", min_value=0, value=0)
    
    cashbackamount = st.number_input("Cashback Amount", min_value=0.0, value=0.0)

    # Create input dataframe
    input_data = {
        
        'gender': [gender],
        'age': [age],
        'city': [city],
        'membershiptype': [membershiptype],
        'itemspurchased': [itemspurchased],
        'dayssincelastpurchase': [dayssincelastpurchase],
        'hourspendonapp': [hourspendonapp],
        'productcategory': [productcategory],
        'productprice': [productprice],
        'quantity': [quantity],
        'ordercount': [ordercount],
        'couponused': [couponused],
        'cashbackamount': [cashbackamount]
    }
    input_df = pd.DataFrame(input_data)

    if st.button("Get Recommendations"):
        distances, indices = recommend_products(pipeline_best_model, input_df)
        
        # Display recommendations
        recommended_product_ids = indices[0]
        recommended_products = data.loc[recommended_product_ids, ['productid', 'description', 'productcategory', 'productprice']]
        
        st.write("Recommended Products:")
        st.write(recommended_products)

if __name__ == "__main__":
    app()
