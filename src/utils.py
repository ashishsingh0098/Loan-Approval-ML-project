import os
import sys
import numpy as np 
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """Save an object to a file using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple models and return a report of their R2 scores."""
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

def model_metrics(true, predicted):
    """Calculate and return various regression metrics."""
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception occurred while evaluating metrics')
        raise CustomException(e, sys)

def print_evaluate_results(X_train, y_train, X_test, y_test, model):
    """Print and log the evaluation results of a model."""
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        model_train_mae, model_train_rmse, model_train_r2 = model_metrics(y_train, y_train_pred)
        model_test_mae, model_test_rmse, model_test_r2 = model_metrics(y_test, y_test_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')

        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logging.info('Exception occurred during printing of evaluated results')
        raise CustomException(e, sys)

def load_object(file_path):
    """Load an object from a file using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
