import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import os
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]
            categorical_columns = ["Married_Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

            preprocessor = ColumnTransformer([
                ("numerical_scaler", StandardScaler(), numerical_columns),
                ("categorical_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Risk_Flag"
            drop_columns = [target_column_name]
                        
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # # Ensure the target arrays are 2D
            # train_arr = target_feature_train_df.values.reshape(-1, 1)
            # test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Concatenate the arrays

            # train_arr = [[input_feature_train_arr,target_feature_train_df]]
            # test_arr =  [[input_feature_test_arr,target_feature_test_df]]

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            # test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
