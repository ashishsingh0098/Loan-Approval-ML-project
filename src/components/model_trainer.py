import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, model_metrics

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(n_estimators = 100, random_state = 21, n_jobs = -1,max_features = 7, max_depth = 30),
                "Decision Tree": DecisionTreeClassifier(max_leaf_nodes=128, random_state = 21),
                "XGBClassifier": XGBClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(algorithm="SAMME")
            }

            for model_name, model in models.items():
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                mae, rmse, r2_square = model_metrics(y_train, y_train_pred)
                print(f"{model_name} Model performance for Training set")
                print("- Root Mean Squared Error: {:.4f}".format(rmse))
                print("- Mean Absolute Error: {:.4f}".format(mae))
                print("- R2 Score: {:.4f}".format(r2_square))

                mae, rmse, r2_square = model_metrics(y_test, y_test_pred)
                print(f"{model_name} Model performance for Test set")
                print("- Root Mean Squared Error: {:.4f}".format(rmse))
                print("- Mean Absolute Error: {:.4f}".format(mae))
                print("- R2 Score: {:.4f}".format(r2_square))
                print('=' * 35)
                print('\n')

            best_model_score = max([model_metrics(y_test, model.predict(X_test))[2] for model in models.values()])
            best_model_name = [list(models.keys())[list(models.values()).index(model)] for model in models.values() if model_metrics(y_test, model.predict(X_test))[2] == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
                        
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            y_test_pred = best_model.predict(X_test)
            mae, rmse, r2 = model_metrics(y_test, y_test_pred)

            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 : {r2}')
            logging.info('Final model training Completed')

            return mae, rmse, r2
        
        except Exception as e:
            raise CustomException(e, sys)
