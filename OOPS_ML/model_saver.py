import pickle
import os
import joblib

class ModelSaver:
    def __init__(self, models):
        self.models = models

    # def save_models(self, save_dir='saved_models'):
    #     # Create the directory if it doesn't exist
    #     os.makedirs(save_dir, exist_ok=True)

    #     # Create separate directories for pickle and joblib
    #     pickle_dir = os.path.join(save_dir, 'pickle_models')
    #     joblib_dir = os.path.join(save_dir, 'joblib_models')
    #     os.makedirs(pickle_dir, exist_ok=True)
    #     os.makedirs(joblib_dir, exist_ok=True)

    #     for name, model in self.models.items():
    #         if model is not None:
    #             # Save with pickle
    #             filename_pickle = os.path.join(pickle_dir, f'{name}_model.pkl')
    #             with open(filename_pickle, 'wb') as file:
    #                 pickle.dump(model, file)
    #             print(f"Model '{name}' saved as '{filename_pickle}'")

    #             # Save with joblib
    #             filename_joblib = os.path.join(joblib_dir, f'{name}_model.joblib')
    #             joblib.dump(model, filename_joblib)
    #             print(f"Model '{name}' saved as '{filename_joblib}'")

    def save_models(self, save_dir='saved_models'):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            if model is not None:
                # Save with pickle
                filename_pickle = os.path.join(save_dir, f'{name}_model.pkl')
                with open(filename_pickle, 'wb') as file:
                    pickle.dump(model, file)
                print(f"Model '{name}' saved as '{filename_pickle}'")

                # Save with joblib
                filename_joblib = os.path.join(save_dir, f'{name}_model.joblib')
                joblib.dump(model, filename_joblib)
                print(f"Model '{name}' saved as '{filename_joblib}'")
    