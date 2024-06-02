# Basic Import
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "K-Neighbors Regressor": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Regressor": RandomForestClassifier(),
                "XGBRegressor": XGBClassifier(),
                "CatBoosting Regressor": CatBoostClassifier(verbose=False),
                "GradientBoosting Regressor": GradientBoostingClassifier(),
                "AdaBoost Regressor": AdaBoostClassifier()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')

            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboost')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info('Model pickle file saved')

            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            y_test_pred = best_model.predict(X_test)

            mae, rmse, r2 = model_metrics(y_test, y_test_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('Final Model Training Completed')

            return mae, rmse, r2

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)