import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)  # Update with your file format if not CSV
        return self.data
