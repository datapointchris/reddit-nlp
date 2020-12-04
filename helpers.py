import sqlite3
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd

# ========================= CLASS LABELS ========================= #

class_labels_all = ["deeplearning", "tensorflow", "scikit_learn", "bigdata", "aws",
                    "awscertifications", "css", "html", "javascript", "shittyprogramming",
                    "java", "sql", "learnsql", "postgresql", "softwarearchitecture", "scala",
                    "apachespark", "mongodb", "linux", "linux4noobs", "datascience", "machinelearning",
                    "etl", "python", "dataengineering"]


def get_random_class_labels(num=8):
    return np.random.choice(class_labels_all, num, replace=False)


# ========================= STOP WORDS ========================= #

useless_words = set(['postgres', 'big', 'panda', 'using', 'scikit', 'sklearn', 'apache', 'spark', 'lambda', 's3',
                     'does', 'looking', 'help', 'new', 'data', 'science', 'scientist', 'machine', 'learning', 'use',
                     'need', 'engineer', 'engineering'])

custom_stop_words = ENGLISH_STOP_WORDS.union(useless_words).union(set(class_labels_all))


def load_sqlite(database, query=None, class_labels=None):

    try:
        connection = sqlite3.connect(database)
    except Exception as e:
        print(f"The error '{e}' occurred connecting")

    placeholders = ','.join('?' for label in class_labels)

    ### FIX ###
    # this query needs to be explicitely given in each notebook
    # to allow for different databases
    subreddit_query = """
    SELECT
        title,
        subreddit,
        date
    FROM subreddits
    WHERE subreddit IN (%s);""" % str(placeholders)

    cursor = connection.cursor()
    cursor.execute(subreddit_query, class_labels)

    column_names = [description[0] for description in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data=data, columns=column_names)
    df = df.drop_duplicates(subset='title')
    for label in class_labels:
        if len(df[df['subreddit'] == label]) == 0:
            raise ValueError(f'No data for "{label}"')
    return df


def print_label_distribution(df, y, labels):
    '''Prints the number of rows per label'''
    for label in labels:
        print(f'{label}: {len(df[y == label])}')
    print()
    print(f'AVERAGE: {int(len(df) / len(labels))}')


def resample_data(X, y, sample_method='max', distribution='concatenate', random_state=None):
    '''Resamples each label to the average number of posts across labels
    Note: Oversample AFTER splitting to train and test set in order to avoid duplicates between
        the train and test set which will give falsely better metrics
    
    Parameters
    ----------
    X : pd.Series
    y : pd.Series
    sample_method : {'max', 'average', 'min', None}
        Method to use to determine the sample numbers.  Methods are calculated between all classes.
    distribution : {'concatenate', 'sample'}
        `concatenate` adds extra samples onto original data
        `resample` resamples the entire dataset, possibly not including all of the original data 
    '''
    if sample_method is None:
        return X, y
    elif sample_method == 'max':
        n_samples = int(np.unique(y, return_counts=True)[1].max())
    elif sample_method == 'average':
        n_samples = int(np.unique(y, return_counts=True)[1].mean())
    elif sample_method == 'min':
        n_samples = int(np.unique(y, return_counts=True)[1].min())

    resampled_X = pd.Series()
    resampled_y = pd.Series()
    np.random.seed(random_state)

    if distribution == 'resample':
        for label in np.unique(y):
            indexes = y[y == label].index
            sampled_indexes = np.random.choice(indexes, size=n_samples, replace=True)
            resampled_X = resampled_X.append(X[sampled_indexes])
            resampled_y = resampled_y.append(y[sampled_indexes])
    elif distribution == 'concatenate':
        for label in np.unique(y):
            indexes = y[y == label].index
            if len(indexes) <= n_samples:
                # Original data
                resampled_X = resampled_X.append(X[indexes])
                resampled_y = resampled_y.append(y[indexes])
                # Add extra samples
                sample_difference = n_samples - len(indexes)
                sampled_indexes = np.random.choice(indexes, size=sample_difference, replace=True)
                resampled_X = resampled_X.append(X[sampled_indexes])
                resampled_y = resampled_y.append(y[sampled_indexes])
            else:
                sampled_indexes = np.random.choice(indexes, size=n_samples, replace=False)
                resampled_X = resampled_X.append(X[sampled_indexes])
                resampled_y = resampled_y.append(y[sampled_indexes])       
    return resampled_X, resampled_y
