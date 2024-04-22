# %% [markdown]
# # Improve Model
# 

# %% [markdown]
# ---

# %%
import joblib
from xgboost import XGBClassifier
from pprint import pprint
from glob import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, cross_val_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import ParameterGrid

import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


%load_ext autoreload
%autoreload 2

# %%
from helpers import load_sqlite, custom_stop_words, get_random_class_labels, resample_data
from visualizer import Visualizer

# %%
random_state = 77

labels = ['python','javascript','html']

df = load_sqlite(database='reddit.sqlite', class_labels=labels)

# %%
imported_model = joblib.load('01_best_model')

# %%
labeler = LabelEncoder()
ada = ADASYN(random_state=random_state)
smote = SMOTE(random_state=random_state)
tfidf = imported_model.named_steps.prep
mnb = imported_model.named_steps.clf

# %%
X = df['title']
y = df['subreddit']

# print('Y_train value counts', y_train.value_counts())
# print('Y_train unique:', np.unique(y_train, return_counts=True))
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)

# %%
grid = [{'resample_ada': [True, False],
         'resample_smote': [True, False],
         'resample_data_method': ['max', 'min', 'average', None],
         'resample_data_distribution': ['concatenate', 'resample']
         }]
param_grid = ParameterGrid(grid)

# %%
models_dict = dict()
best_score = 0.0
best_model = None

# %%
for i, param in enumerate(param_grid):
    print('*'*50)
    print('PARAMS:')
    print(param)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    
    X_train, y_train = resample_data(X_train, y_train,
                                     sample_method=param['resample_data_method'],
                                     distribution=param['resample_data_distribution'],
                                     random_state=random_state)

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    y_train = labeler.fit_transform(y_train)
    y_test = labeler.transform(y_test)

    if param['resample_ada'] is True:
        X_train, y_train = ada.fit_resample(X_train, y_train)

    if param['resample_smote'] is True:
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scores = cross_val_score(mnb, X_train, y_train)
    mean_score = np.mean(scores)
    print('Scores: ', scores)
    print('Mean Score: ', mean_score)
    print('*'*50)
    print()
    print()
    
    param['score'] = mean_score
    models_dict[i] = param
    if mean_score > best_score:
        best_score = mean_score
        best_model = param
    

# %%
print(best_score)
print(best_model)

# %%
pd.DataFrame(models_dict).T.sort_values('score', ascending=False)

# %%



