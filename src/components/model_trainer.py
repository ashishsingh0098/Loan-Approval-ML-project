import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, model_metrics

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            if train_array.shape[0] == 0 or test_array.shape[0] == 0:
                raise CustomException("Input arrays are empty")

            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBClassifier": XGBClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(algorithm="SAMME")
            }

            best_model_score = -1
            best_model_name = None
            best_model = None

            for model_name, model in models.items():
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                logging.info(f"{model_name} Model performance for Training set")
                logging.info(f"- Score: {train_score:.4f}")

                logging.info(f"{model_name} Model performance for Test set")
                logging.info(f"- Score: {test_score:.4f}")
                logging.info('=' * 35)

                if test_score > best_model_score:
                    best_model_score = test_score
                    best_model_name = model_name
                    best_model = model

            if best_model_score < 0.6:
                raise CustomException("No best model found")

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

            logging.info(f'Test Accuracy : {accuracy_score(y_test, y_test_pred):.4f}')
            logging.info(f'Test Precision : {precision_score(y_test, y_test_pred):.4f}')
            logging.info(f'Test Recall : {recall_score(y_test, y_test_pred):.4f}')
            logging.info(f'Test F1 Score : {f1_score(y_test, y_test_pred):.4f}')
            logging.info('Final model training Completed')

            return mae, rmse, r2
        
        except Exception as e:
            raise CustomException(e, sys)
