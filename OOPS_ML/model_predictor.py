class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
