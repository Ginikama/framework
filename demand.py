import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import joblib

# Custom ARIMA model wrapper to integrate with scikit-learn pipeline
class ARIMAWrap(BaseEstimator, RegressorMixin):
    def __init__(self, order=(3, 2, 1)):
        self.order = order
        self.model_ = None

    def fit(self, X, y):
        self.model_ = ARIMA(y, order=self.order).fit()
        return self

    def predict(self, X):
        # Forecast for the length of X
        return self.model_.forecast(steps=len(X))

# Function to create and save the demand forecasting pipeline
def create_demand_pipeline():
    # Load your dataset
    demand_df = pd.read_csv('cleaned_data.csv')

    # Select relevant columns for demand forecasting
    columns = ['timestamp', 'productcategory', 'quantity']
    demand_df = demand_df[columns]

    # Convert 'timestamp' to datetime with a specified format
    demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'], errors='coerce')

    # Extract date features from 'timestamp'
    demand_df['year'] = demand_df['timestamp'].dt.year
    demand_df['month'] = demand_df['timestamp'].dt.month
    demand_df['day'] = demand_df['timestamp'].dt.day

    demand_df['year'] = demand_df['year'].map('{:g}'.format)
    demand_df['month'] = demand_df['month'].map('{:g}'.format)
    demand_df['day'] = demand_df['day'].map('{:g}'.format)

    # Group by date and product category, aggregating the quantity
    demand_df = demand_df.groupby(['year', 'month', 'day', 'productcategory']).agg({'quantity': 'sum'}).reset_index()

    # Define categorical and numerical columns
    categorical_columns = ['productcategory']
    numerical_columns = ['year', 'month', 'day']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_columns),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns)
        ]
    )

    # Split the data into features and target
    X = demand_df.drop(columns=['quantity'])
    y = demand_df['quantity']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('arima', ARIMAWrap(order=(3, 2, 1)))  # Using the optimized ARIMA parameters
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate MAE
    arima_mae = mean_absolute_error(y_test, y_pred)
    print("ARIMA MAE:", arima_mae)

    # Save the pipeline
    joblib.dump(pipeline, 'demand_pipeline.joblib')

# Run the function to create and save the pipeline
create_demand_pipeline()
