import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        # Perform feature engineering steps
        
        # Example: Scaling numerical features
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        # Add new features or transformations
        
        # Example: Creating a new feature based on existing ones
        data['new_feature'] = data['feature1'] * data['feature2']
        
        # Drop irrelevant or redundant features
        
        # Example: Dropping a feature
        data.drop(columns=['irrelevant_feature'], inplace=True)
        
        return data
