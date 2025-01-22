import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.util import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and testing input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient boosting":GradientBoostingRegressor(),
                "Linear Regressor":LinearRegression(),
                "k nearest classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "Adaboost classifier":AdaBoostRegressor()
            }

            params={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64]
                },
                'Gradient Boosting':{
                    'learning_rate':[0.1,0.01,0.05]
                },
                "Linear Regression":{},
                'k-Neighbour Regressor':{
                    'n_neighbors':[5,7,9,11]
                },
                'XGBRegressor':{
                    'n_estimators':[8,16,24,32]
                },
                'CaeBoosting Regressor':{
                    'depth':[6,8,10],
                    'iteration':[30,50,100]
                },
                'Adaboost Regressor':{
                    'learning_rate':[0.1,0.01,0.05]
                }


            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)