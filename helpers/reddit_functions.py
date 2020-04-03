import datetime
import json
import time
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import wordcloud
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from grid_models import preprocessors, estimators

# use this class and functions the same way as StandardScaler is used.
# Use the class to find the functions
# call the functions to modify data assigned to a variable.

def function_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        return_value = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        try:
            logger.info(f'Elapsed time: {round(elapsed_time/60,2)} minutes for {func.__name__}')
        except:
            print(f"Elapsed time: {round(elapsed_time/60,2)} minutes for function: '{func.__name__}'")
        return return_value
    return wrapper


class Reddit:

    def __init__(self):
        pass


    def compare_models(self, X_train, X_test, y_train, y_test, preprocessors=preprocessors, estimators=estimators, subs=None, cv=5, verbose=1, n_jobs=-1):

        # Set up the dataframe
        model_comparison_df = pd.DataFrame(columns=[
                                                'date',
                                                'preprocessor',
                                                'estimator',
                                                'best_params', 
                                                'best_train_score',
                                                'best_test_score',
                                                'variance',
                                                'prep_code',
                                                'est_code',
                                                'sub_list'
                                                ])         
        # fit a model for each combo of preprocessor and estimator
        for est in estimators.values():
            for prep in preprocessors.values():
                print(f"Fitting model with {prep['name']} and {est['name']}")
                pipe = Pipeline([(prep['abbr'], prep['processor']), (est['abbr'], est['estimator'])])
                pipe_params = dict()
                pipe_params.update(prep['pipe_params'])
                pipe_params.update(est['pipe_params'])
                model = GridSearchCV(pipe, param_grid=pipe_params, cv=cv, verbose=verbose, n_jobs=n_jobs)
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                now = datetime.datetime.now()
                subreddits = (', ').join(subs) if subs is not None else 'na'

                # add the model result to the df
                model_comparison_df.loc[len(model_comparison_df)] = [
                                        now,
                                        prep['name'],
                                        est['name'], 
                                        model.best_params_, 
                                        train_score,
                                        test_score,
                                        (train_score - test_score) / train_score * 100,
                                        prep['abbr'],
                                        est['abbr'],
                                        subreddits
                                    ]

        return model_comparison_df
