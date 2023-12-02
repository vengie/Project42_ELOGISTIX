import pandas as pd
from data_loader import DataLoader 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_preprocessor import DataPreprocessor 
from model_trainer import ModelTrainer
from model_saver import ModelSaver
from model_loader import ModelLoader
from model_evaluator import ModelEvaluator
from model_predictor import ModelPredictor
from hyperparameter_tuner import SVMHyperparameterTuning
from model_performance_fine_tuning import ModelPerformanceFineTuning

def main():
    file_path = "https://raw.githubusercontent.com/vengie/Project42_ELOGISTIX/main/Data/DSMMProject42-CPL-5559-Ecom_Shipping_stride.csv"  
    data = pd.read_csv(file_path)


    #Load Data
    if data is not None:
        print(data.head())

    # Preprocess Data including feature engineering
    # Initialize the DataPreprocessor and preprocess the data
    preprocessor = DataPreprocessor()
    transformed_data = preprocessor.fit_transform(data)

    # Split the data into features and target
    X = transformed_data.drop(columns=['Reached.on.Time_Y.N'])
    y = transformed_data['Reached.on.Time_Y.N']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training 
    model_trainer = ModelTrainer()  
    model_trainer.preprocess_data(X_train, X_test, y_train, y_test)  
    model_trainer.train_models()

   
    # Savetrained models into pkl and joblib format
    model_saver = ModelSaver(models=model_trainer.models)

    # Save the models
    model_saver.save_models()

    # Create an instance of ModelLoader
    loader = ModelLoader()
    # Load models from the default directory
    loader.load_models()
    

    # Model Evaluation
    # Create an instance of ModelEvaluator and pass the loaded models
    evaluator = ModelEvaluator(models=loader.models)

    # Evaluate the loaded models
    eval_results = evaluator.evaluate_models(X_test, y_test)

    # Identify the best model
    best_model_name = max(eval_results, key=lambda k: eval_results[k]['accuracy'])
    print(f"The best model is: {best_model_name}")

    # Use the best model for prediction
    best_model = loader.get_model(best_model_name)
    if best_model is not None:
        predictions = best_model.predict(X_test)
        print(f"Predictions using {best_model_name} the best model")
        print(predictions)

    # Some new data for prediction
    new_data = pd.read_csv('https://raw.githubusercontent.com/vengie/Project42_ELOGISTIX/main/Data/new_data.csv' )  

    # preprocessor = DataPreprocessor()
    transformed_new_data = preprocessor.transform(new_data)
    new_data_without_target = transformed_new_data.drop(columns=['Reached.on.Time_Y.N'])
    

    # After loading models and evaluator
    predictor = ModelPredictor(best_model)

    # Assuming 'new_data' is the DataFrame with your new data
    predictions = predictor.predict(input_data=new_data_without_target)
    print(predictions)



    #Model Performance Finetune
    model_tuner = ModelPerformanceFineTuning()
    model_tuner.fine_tune_performance()
 
    # Generate and display the evaluation reports for the fine-tuned models
    model_tuner.generate_model_reports()

if __name__ == "__main__":
    main()