from sklearn.metrics import classification_report

class ModelEvaluator:
    def __init__(self, models):
        self.models = models
        self.evaluation_results_dict = {}  # Initialize the evaluation results dictionary

    

    def evaluate_models(self, X_test, y_test):
        evaluation_results_dict = {}
        for name, model in self.models.items():
            if model is not None:
                predictions = model.predict(X_test)
                report = classification_report(y_test, predictions, output_dict=True)
                evaluation_results_dict[name] = {
                    'accuracy': report['accuracy'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1-score': report['weighted avg']['f1-score']
                }
                print(f"Evaluation Report for {name}:")
                print(classification_report(y_test, predictions))
        return evaluation_results_dict
    
    # def evaluate_models(self, X_test, y_test):
    #     for name, model in self.models.items():
    #         if model is not None:  # Check if the model exists
    #             try:
    #                 predictions = model.predict(X_test)
    #                 report = classification_report(y_test, predictions, output_dict=True)
    #                 self.eval_results[name] = {
    #                     'accuracy': report['accuracy'],
    #                     'precision': report['weighted avg']['precision'],
    #                     'recall': report['weighted avg']['recall'],
    #                     'f1-score': report['weighted avg']['f1-score']
    #                 }
    #                 print(f"Evaluation Report for {name}:")
    #                 print(classification_report(y_test, predictions))
    #             except AttributeError as e:
    #                 print(f"Error evaluating {name}: {e}")
    #         else:
    #             print(f"Model '{name}' is None. Skipping evaluation.")
        
    #     return self.eval_results
