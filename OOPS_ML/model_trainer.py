import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


class ModelTrainer:
    # def __init__(self, data):
    #     self.data = data
    def __init__(self):
        # Initialize any necessary attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Initialize models
        self.logreg = LogisticRegression(max_iter=1000)
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.xgb = XGBClassifier(random_state=42)
        self.svm = SVC(random_state=42)

    def preprocess_data(self, X_train, X_test, y_train, y_test):
        # Preprocess the data
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # Example: Encoding categorical columns
        le = LabelEncoder()
        cat_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
        for col in cat_cols:
            self.X_train[col] = le.fit_transform(self.X_train[col])
            self.X_test[col] = le.transform(self.X_test[col])


        
    def train_models(self):
        # Initialize models
        self.logreg = LogisticRegression(max_iter=1000)
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.xgb = XGBClassifier(random_state=42)
        self.svm = SVC(random_state=42)
        
        # Train models
        self.logreg.fit(self.X_train, self.y_train)
        self.decision_tree.fit(self.X_train, self.y_train)
        self.xgb.fit(self.X_train, self.y_train)
        self.svm.fit(self.X_train, self.y_train)

        # Store trained models in a dictionary
        self.models = {
            'logreg': self.logreg,
            'decision_tree': self.decision_tree,
            'xgb': self.xgb,
            'svm': self.svm
        }
    


    # def evaluate_models(self):
    #     self.train_models()  # Ensure models are trained before evaluation

    #     # Evaluate models
    #     models = {
    #         'Logistic Regression': self.logreg,
    #         'Decision Tree': self.decision_tree,
    #         'XGBoost': self.xgb,
    #         'SVM': self.svm
    #     }

    #     for name, model in models.items():
    #         model.fit(self.X_train, self.y_train)  # Fit the model here
    #         predictions = model.predict(self.X_test)
    #         report = classification_report(self.y_test, predictions)
    #         print(f"Classification Report for {name}:")
    #         print(report)