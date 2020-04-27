import argparse
import datetime
import functools
import json
import logging
import logging.handlers
import os
import sys
import time

from sklearn.model_selection import train_test_split

# set path to current working directory for cron job
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# path hack, I know it's gross
sys.path.insert(0, os.path.abspath('..'))

from util import dataloader, grid_models
from util.reddit_functions import Labeler, Reddit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
file_handler = logging.handlers.RotatingFileHandler(filename='../logs/compare_models.log', maxBytes=10000000, backupCount=10)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def function_timer(func):
    """Prints runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f'Elapsed time: {round(elapsed_time/60,2)} minutes for {repr(func.__name__)}')
        else:
            print(f"Elapsed time: {round(elapsed_time/60,2)} minutes for function: '{repr(func.__name__)}'")
        return value
    return wrapper


@function_timer
def main():
    subreddit_list = ['css', 'html', 'machinelearning', 'python']

    df = dataloader.data_selector(subreddit_list, 'sqlite')

    X = df['title']
    y = df['subreddit']

    labeler = Labeler()
    labeler.fit(y)
    y = labeler.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

    date = str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M'))
    preprocessors = grid_models.preprocessors
    estimators = grid_models.estimators
    model = Reddit()

    logger.info(f'Training models')
    try:
        compare_df = model.compare_models(X_train, X_test, y_train, y_test,
                                          preprocessors=preprocessors,
                                          estimators=estimators,
                                          classes=subreddit_list,
                                          cv=3,
                                          verbose=1)
    except Exception:
        # I know this is bad, will change it when I convert to airflow
        logger.exception(f'Error comparing models:')

    logger.info(f'Saving comparison df')
    try:
        compare_df.to_csv(f'../data/compare_df/{date}.csv')
    except FileNotFoundError:
        logger.exception('Path is wrong')

if __name__ == "__main__":
    logger.info('PROGRAM STARTED -- "compare_models"')
    main()
    logger.info('PROGRAM FINISHED')
