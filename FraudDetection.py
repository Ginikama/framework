import streamlit as st
import pandas as pd
import joblib

# Load the saved model pipeline
pipeline = joblib.load('fraud_detection_pipeline.joblib')

def app():
    st.title('Fraud Detection')

    # Create input fields for each feature
    age = st.slider('Age', min_value=0, max_value=100, value=30)
    totalspend = st.number_input('Total Spend', min_value=0.0, value=100.0)
    itemspurchased = st.number_input('Items Purchased', min_value=1, value=1)
    productprice = st.number_input('Product Price', min_value=0.0, value=20.0)
    totalpurchaseamount = st.number_input('Total Purchase Amount', min_value=0.0, value=150.0)
    couponused = st.number_input('Coupon Used', min_value=0, value=0)
    cashbackamount = st.number_input('Cashback Amount', min_value=0.0, value=5.0)
    ordercount = st.number_input('Order Count', min_value=1, value=1)
    dayssincelastpurchase = st.slider('Days Since Last Purchase', min_value=0, max_value=365, value=30)
    daysincelastorder = st.slider('Days Since Last Order', min_value=0, max_value=365, value=10)
    quantity = st.number_input('Quantity', min_value=1, value=1)
    unitprice = st.number_input('Unit Price', min_value=0.0, value=10.0)
    hourspendonapp = st.number_input('Hours Spent on App', min_value=0.0, value=1.0)
    numberofdeviceregistered = st.number_input('Number of Devices Registered', min_value=1, value=1)
    complain = st.checkbox('Complain', value=False)
    churn = st.checkbox('Churn', value=False)
    tenure = st.number_input('Tenure (Months)', min_value=0, value=12)

    # Select box for categorical features
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    city = st.selectbox('City', options=['New York', 'Miami', 'San Francisco', 'Los Angeles', 'Houston'])
    membershiptype = st.selectbox('Membership Type', options=['Silver', 'Gold'])
    paymentmethod = st.selectbox('Payment Method', options=['Credit Card', 'PayPal'])
    shipmode = st.selectbox('Ship Mode', options=['Standard Class', 'Second Class', 'Same Day'])

    # Button to trigger prediction
    if st.button('Detect'):
        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'age': [age],
            'totalspend': [totalspend],
            'itemspurchased': [itemspurchased],
            'productprice': [productprice],
            'totalpurchaseamount': [totalpurchaseamount],
            'couponused': [couponused],
            'cashbackamount': [cashbackamount],
            'ordercount': [ordercount],
            'dayssincelastpurchase': [dayssincelastpurchase],
            'daysincelastorder': [daysincelastorder],
            'quantity': [quantity],
            'unitprice': [unitprice],
            'hourspendonapp': [hourspendonapp],
            'numberofdeviceregistered': [numberofdeviceregistered],
            'complain': [1 if complain else 0],
            'churn': [1 if churn else 0],
            'tenure': [tenure],
            'gender': [gender],
            'city': [city],
            'membershiptype': [membershiptype],
            'paymentmethod': [paymentmethod],
            'shipmode': [shipmode]
        })

        # Predict with the pipeline
        prediction = pipeline.predict(input_data)
        prediction_proba = pipeline.predict_proba(input_data)[:, 1]

        # Display the prediction and probability
        if prediction[0] == 1:
            st.write(f"**Prediction:** Fraudulent")
        else:
            st.write(f"**Prediction:** Not Fraudulent")
        
        st.write(f"**Probability of Fraud:** {prediction_proba[0]:.2%}")
