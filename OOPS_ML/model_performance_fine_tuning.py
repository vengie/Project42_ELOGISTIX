from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class ModelPerformanceFineTuning:
    def __init__(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.logreg = LogisticRegression()
        self.dt = DecisionTreeClassifier()
        self.xgb = GradientBoostingClassifier()
        self.svm = SVC()
        self.best_model = None
        self.best_model_name = None
        self.best_model_accuracy = 0.0

    def fine_tune_performance(self):
        # Logistic Regression
        self.logreg_params = {'C': [0.1, 1, 10], 'penalty': ['l2', 'none']}
        self.logreg = self._tune_model(self.logreg, self.logreg_params)

        # Decision Tree
        self.dt_params = {'max_depth': [None, 5, 10, 20]}
        self.dt = self._tune_model(self.dt, self.dt_params)

        # XGBoost
        xgb_params = {'learning_rate': [0.1, 0.01], 'n_estimators': [100, 200, 300]}
        self.xgb = self._tune_model(self.xgb, xgb_params)

        # SVM
        svm_params = {'C': [1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf', 'linear']}
        self.svm = self._tune_model(self.svm, svm_params)

    def _tune_model(self, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for {type(model).__name__}:", grid_search.best_params_)
        print(f"Best Estimator for {type(model).__name__}:", best_model)
        return best_model

    def _evaluate_and_track_best(self, model, model_name):
        predictions = model.predict(self.X_test)
        report = classification_report(self.y_test, predictions, output_dict=True)
        print(f"Evaluation Report for {model_name}:")
        print(classification_report(self.y_test, predictions))

        accuracy = report['accuracy']
        if self.best_model is None or accuracy > self.best_model_accuracy:
            self.best_model = model
            self.best_model_name = model_name
            self.best_model_accuracy = accuracy

    def generate_model_reports(self):
        self._evaluate_and_track_best(self.logreg, 'Logistic Regression')
        self._evaluate_and_track_best(self.dt, 'Decision Tree')
        self._evaluate_and_track_best(self.xgb, 'XGBoost')
        self._evaluate_and_track_best(self.svm, 'SVM')

        print(f"Best Model: {self.best_model_name}")
        print("Evaluation Report for Best Model:")
        best_model_predictions = self.best_model.predict(self.X_test)
        print(classification_report(self.y_test, best_model_predictions))