import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
import joblib
from sklearn.preprocessing import StandardScaler


selected_columns = [
    'totalspend', 
    'age', 
    'satisfactionscore', 
    'city', 
    'membershiptype', 
    'dayssincelastpurchase', 
    'discountapplied', 
    'ordercount'
]

# Define custom transformers
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

class EncodeCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Apply one-hot encoding to categorical columns
        return pd.get_dummies(X, columns=['city', 'membershiptype'], drop_first=True)

class StandardizeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        numerical_features = ['totalspend', 'age', 'satisfactionscore', 'dayssincelastpurchase', 'ordercount']
        self.scaler.fit(X[numerical_features])
        return self
    
    def transform(self, X):
        numerical_features = ['totalspend', 'age', 'satisfactionscore', 'dayssincelastpurchase', 'ordercount']
        X[numerical_features] = self.scaler.transform(X[numerical_features])
        return X

# Define the preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('feature_selector', FeatureSelector(selected_columns)),
    ('categorical_encoder', EncodeCategorical()),
    ('standardizer', StandardizeFeatures())
])

# Define the complete segmentation pipeline
segment_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessing_pipeline),
    ('clustering', AgglomerativeClustering(n_clusters=7, linkage='ward'))
])

# Save the pipeline
joblib.dump(segment_pipeline, 'segment_pipeline.pkl')
