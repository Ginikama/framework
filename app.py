import streamlit as st
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
import Segmentation
import Recommendation
import FraudDetection
import DemandForecasting

def landing_page(logo):
    st.title('E-commerce Analytics')
    st.image(logo, width=250)
    st.write("""
    Welcome to the E-commerce Analytics application. This tool provides various analytics features to help you understand and improve your e-commerce business.
    
    - **Customer Segmentation**: Analyze and segment your customers based on different criterias.
    - **Fraud Detection**: Identify potentially fraudulent transactions and protect your business.
    - **Personalized Recommendation**: Offer tailored product recommendations to enhance customer satisfaction.
    - **Demand Forecasting**: Predict future product demand to optimize inventory management.
    
    Select a task from the sidebar to get started.
    """)

def run_task(selected_task):
    if selected_task == 'Customer Segmentation':
        st.header('Customer Segmentation')
        # Show file upload widget only for Customer Segmentation task
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            Segmentation.app(df)
    elif selected_task == 'Fraud Detection':
        st.header('Fraud Detection')
        FraudDetection.app()
    elif selected_task == 'Personalized Recommendation':
        st.header('Personalized Recommendation')
        Recommendation.app()
    elif selected_task == 'Demand Forecasting':
        st.header('Demand Forecasting')
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            DemandForecasting.app(df)

def main():
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    logo = current_dir / "Photos" / "logo.png"
    logo = Image.open(logo)

    st.set_page_config(
        page_title="E-commerce Analytics",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    style = {
        "nav-link": {"font-family": "Monospace, Arial", "--hover-color": "rgb(255, 192, 0)"},
        "nav-link-selected": {"background-color": "rgb(204, 85, 0)", "font-family": "Monospace , Arial"},
    }

    with st.sidebar:
        st.title('E-commerce Data Analytics')
        st.image(logo, width=175)
        app = option_menu(None,
                          options=['Home', 'Customer Segmentation', 'Fraud Detection', 'Demand Forecasting', 'Personalized Recommendation'],
                          icons=["house-door", "pie-chart-fill", "kanban", "person-check-fill"],
                          styles=style,
                          default_index=0,
                          )

    if app == 'Home':
        landing_page(logo)
    else:
        run_task(app)

if __name__ == '__main__':
    main()
