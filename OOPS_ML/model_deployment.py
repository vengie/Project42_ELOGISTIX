from model_loader import ModelLoader
from model_evaluator import ModelEvaluator
from model_predictor import ModelPredictor


class ModelDeployment:
    def __init__(self):
        self.loader = ModelLoader()
        self.evaluator = None
        self.loaded_models = None

    def load_models(self, models_dir='saved_models'):
        self.loader.load_models(models_dir)
        self.loaded_models = self.loader.models

    def evaluate_models(self, X_test, y_test):
        if self.loaded_models is None:
            print("No models loaded. Load models first.")
            return
        
        self.evaluator = ModelEvaluator(self.loaded_models)
        eval_results = self.evaluator.evaluate_models(X_test, y_test)
        return eval_results

    def predict_with_best_model(self, new_data):
        if self.loaded_models is None:
            print("No models loaded. Load models first.")
            return

        if self.evaluator is None:
            print("Models not evaluated. Evaluate models first.")
            return

        best_model_name = max(self.evaluator.eval_results,
                              key=lambda k: self.evaluator.eval_results[k]['accuracy'])
        best_model = self.loader.get_model(best_model_name)

        if best_model is not None and hasattr(best_model, 'predict'):
            predictor = ModelPredictor(best_model)
            predictions = predictor.predict(input_data=new_data)
            return predictions
        else:
            print("Best model couldn't be used for prediction or no evaluation results.")
            return None
