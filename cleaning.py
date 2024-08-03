import joblib
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CleanColumnNames(BaseEstimator, TransformerMixin):
    def clean_name(self, name):
        # Remove special characters except underscores and spaces
        name = re.sub(r'[^\w\s]', '', name)
        
        # Replace spaces and underscores with nothing and capitalize following letters
        name = re.sub(r'[_\s]+([a-zA-Z])', lambda x: x.group(1).upper(), name)
        
        # Convert to lowercase
        name = name.lower()
        
        return name

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Apply the cleaning function to all column names
        cleaned_columns = [self.clean_name(col) for col in df.columns]
        
        # Remove trailing numbers from column names
        cleaned_columns = [re.sub(r'(\d+)$', '', col).strip() for col in cleaned_columns]
        
        # Assign cleaned column names to DataFrame
        df.columns = cleaned_columns
        
        # Remove columns that are unnamed
        df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Remove any rows with null values
        df = df.dropna()
        
        return df

# Create the pipeline
pipeline = Pipeline([
    ('clean_columns', CleanColumnNames())
])

# Export the pipeline
joblib.dump(pipeline, 'data_cleaning_pipeline.pkl')