import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

class ModelTrainingConfig:
    def __init__(self):
        self.trained_model_config_path = os.path.join('artifacts', 'model.pickle') 

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('split training and test input data')
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            ) 

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(x_train=x_train, y_train= y_train, x_test=x_test, y_test=y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on both training and testing")

            save_object(filepath=self.model_trainer_config.trained_model_config_path, obj=best_model)

            y_pred = best_model.predict(x_test)
            r2_square = r2_score(y_test, y_pred)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
