import sys
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pickel")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            self.numerical_cols = ["writing_score", "reading_score"]
            self.categorical_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            self.numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )  

            self.categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            ) 

            logging.info("Numerical columns: {self.numerical_cols}")
            logging.info("Categorical columns: {self.categorical_cols}")

            self.preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", self.numerical_pipeline, self.numerical_cols),
                    ("categorical_pipeline", self.categorical_pipeline, self.categorical_cols)
                ]
            )

            logging.info("Numerical columns scaling is complete")
            logging.info("Categorical columns encoding is complete")

            return self.preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info("obtaining processing object")

            target_col_names = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_col_names], axis = 1)
            target_feature_train_df = train_df[target_col_names]

            input_feature_test_df = test_df.drop(columns=[target_col_names], axis = 1)
            target_feature_test_df = test_df[target_col_names]

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("preprocessed the train and test datasets")

            save_object(filepath = self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj)

            logging.info("saved the preprocessed model")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)