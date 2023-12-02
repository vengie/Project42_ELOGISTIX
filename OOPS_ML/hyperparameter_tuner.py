from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

class SVMHyperparameterTuning:
    def __init__(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear', 'poly']
        }
        self.grid_search = None
        self.best_estimator = None

    def tune_hyperparameters(self):
        svm = SVC()
        self.grid_search = GridSearchCV(svm, self.param_grid, cv=5, scoring='accuracy')
        self.grid_search.fit(self.X, self.y)
        self.best_estimator = self.grid_search.best_estimator_
        print("Best Parameters:", self.grid_search.best_params_)
        print("Best Estimator:", self.best_estimator)

    def get_best_model(self):
        return self.best_estimator

