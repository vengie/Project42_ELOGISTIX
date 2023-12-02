import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}

    def fit(self, data):
        # Assuming the data is a pandas DataFrame
        for column in data.columns:
            if data[column].dtype == 'object':
                le = LabelEncoder()
                le.fit(data[column])
                self.label_encoders[column] = le

    def transform(self, data):
        transformed_data = data.copy()
        for column, le in self.label_encoders.items():
            transformed_data[column] = le.transform(data[column])
        return transformed_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)