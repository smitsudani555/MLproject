import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the object to the file
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
# def evaluate_model(x_train,y_train,x_test,y_test,models):
#     try:
#         report={}

#         for i in range(len(list(models))):
#             model=list(models.values())[i]

#             model.fit(x_train,y_train)

#             y_train_pred=model.predict(x_train)
#             y_test_pred=model.predict(x_test)

#             train_model_score=r2_score(y_train,y_train_pred)
#             test_model_score=r2_score(y_test,y_test_pred)

#             report[list(model.keys())[i]]=test_model_score
            
#         return report
def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    """
    Evaluates multiple models and returns their test r2 scores.
    """
    try:
        report = {}

        # Loop through the models dictionary
        for model_name, model in models.items():
            # Get hyperparameters for the model
            para = param.get(model_name, {})

            if para:  # Perform GridSearchCV only if parameters are provided
                gs = GridSearchCV(estimator=model, param_grid=para, cv=3, scoring='r2', verbose=2)
                gs.fit(x_train, y_train)

                # Update model with best parameters
                model = gs.best_estimator_

            # Train the model
            model.fit(x_train, y_train)

            # Predict on training and testing data
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculate r2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save the test model score to the report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

