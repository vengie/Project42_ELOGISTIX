import pickle
import os
from joblib import load

class ModelLoader:
    def __init__(self):
        self.models = {}

    def load_models(self, models_dir='saved_models'):
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"No such directory: {models_dir}")

        for filename in os.listdir(models_dir):
            model_name, ext = os.path.splitext(filename)
            if ext == ".pkl":
                with open(os.path.join(models_dir, filename), 'rb') as file:
                    model = pickle.load(file)
                    self.models[model_name] = model
            elif ext == ".joblib":
                model = load(os.path.join(models_dir, filename))
                self.models[model_name] = model

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"No such model found: {model_name}")

    # def load_models(self, models_dir='saved_models'):
    #     if not os.path.exists(models_dir):
    #         raise FileNotFoundError(f"No such directory: {models_dir}")

    #     for filename in os.listdir(models_dir):
    #         if filename.endswith(".pkl"):
    #             model_name = filename.split("_")[0]
    #             with open(os.path.join(models_dir, filename), 'rb') as file:
    #                 model = pickle.load(file)
    #                 self.models[model_name] = model

    # def get_model(self, model_name):
    #     if model_name in self.models:
    #         return self.models[model_name]
    #     else:
    #         raise ValueError(f"No such model found: {model_name}")
