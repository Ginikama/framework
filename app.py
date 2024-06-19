import streamlit as st
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
import Segmentation
import Recommendation
import FraudDectection
import DemandForecasting

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def landing_page(logo):
    st.title('E-commerce Analytics')
    st.image(logo, width=250)
    st.header('Upload CSV File')
    uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])
    return uploaded_file

def run_task(selected_task, df):
    if selected_task == 'Customer Segmentation':
        Segmentation.app(df)
    elif selected_task == 'Personalized Recommendation':
        Recommendation.app(df)
    elif selected_task == 'Fraud Detection':
        FraudDectection.app(df)
    elif selected_task == 'Demand Forecasting':
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

    uploaded_file = landing_page(logo)
    if uploaded_file:
        df = load_data(uploaded_file)
        style = {
            "nav-link": {"font-family": "Monospace, Arial", "--hover-color": "rgb(255, 192, 0)"},
            "nav-link-selected": {"background-color": "rgb(204, 85, 0)", "font-family": "Monospace , Arial"},
        }

        with st.sidebar:
            st.title('E-commerce Data Analytics')
            st.image(logo, width=175)
            app = option_menu(None,
                              options=['Customer Segmentation', 'Personalized Recommendation', 'Fraud Detection', 'Demand Forecasting'],
                              icons=["pie-chart-fill", "kanban", "person-check-fill"],
                              styles=style,
                              default_index=0,
                              )
        run_task(app, df)

if __name__ == '__main__':
    main()