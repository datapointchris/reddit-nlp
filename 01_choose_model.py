# %% [markdown]
# # Find the Best Model
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
import sklearn

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
def build_model(preprocessor, classifier, cv=3, scoring='roc_auc_ovr', verbose=1):
    '''
    Takes a dictionary with params and outputs a gridsearch model
    '''
    pipe = Pipeline(
    [('prep', preprocessor.get('preprocessor')),
     ('clf', classifier.get('estimator'))])
    
    pipe_params = dict()
    pipe_params.update(preprocessor.get('params'))
    pipe_params.update(classifier.get('params'))
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    model = GridSearchCV(pipe, param_grid=pipe_params, cv=skf, verbose=verbose, scoring=scoring, n_jobs=-1)
    return model

# %%
tfidf = {
    'preprocessor': TfidfVectorizer(stop_words=custom_stop_words),
    'name': 'TF-IDF Vectorizer',
    'params': {
        "prep__ngram_range": [(1, 2)],
        "prep__max_df": [.9],
        "prep__use_idf": [True],
        "prep__norm": ["l2"],
        # "prep__strip_accents": [None, 'ascii', 'unicode'],
        # "prep__ngram_range": [(1, 1), (1, 2)],
        # "prep__max_features": [5000, 6000, 7000],
        # "prep__min_df": np.arange(2, 20, 4),
        # "prep__max_df": np.linspace(.8, .99, 5),
        # "prep__norm": ("l1", "l2"),
        # "prep__use_idf": [True, False]
    }
}

# %%
estimators = {
    "logisticregression": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(max_iter=1000, fit_intercept=False, C=.99),
        "params": {
            "clf__solver": ["lbfgs", "saga"]
        }
    },
    "randomforestclassifier": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(min_samples_leaf=2, min_samples_split=.01),
        "params": {
            "clf__n_estimators": [300, 500, 1000],
            "clf__max_depth": np.linspace(400, 1000, 5, dtype=int)
        }
    },
    "multinomialnb": {
        "name": "Multinomial Bayes Classifier",
        "estimator": MultinomialNB(alpha=.1189),
        "params": {
            "clf__fit_prior": [True, False]
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "estimator": SVC(kernel="sigmoid", probability=True),
        "params": {
            "clf__C": [.99, 1]
        }
    },
    "sgdclassifier": {
        "name": "Stochastic Gradient Descent Classifier",
        "estimator": SGDClassifier(alpha=.0001, fit_intercept=True, penalty="l2", loss="modified_huber"),
        "params":
            {
        }
    },
    'xgbclassifier': {
        'name': 'XGBoost Classifier',
        'estimator': XGBClassifier(n_estimators=200),
        'params': {
            "clf__max_depth": [3, 5, 10],
            # "clf__learning_rate": np.linspace(.001, .1, 3),
            # "clf__n_estimators": [50, 100, 200],
            # "clf__objective": ['binary:logistic', 'multi:softprob'],
            # "clf__booster": ['gbtree', 'gblinear', 'dart'],
            # "clf__gamma": np.linspace(0, 1, 3),
            # "clf__subsample": [.5, 1],
            # "clf__reg_lambda": np.linspace(0, 1, 3),
        }
    }
}

# %%
random_state = 77

labels = ['python','javascript','html']

df = load_sqlite(database='reddit.sqlite', class_labels=labels)

# %%
X = df['title']
y = df['subreddit']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

# %%
fitted_models = dict()
best_auc_score = 0.0
best_model = None

for name, estimator in estimators.items():
    print("*"*50)
    print(f'Model: {estimator.get("name")}')
    print()
    model = build_model(preprocessor=tfidf, classifier=estimator, cv=5, verbose=0)
    model.fit(X_train, y_train)
    print()
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f'Train Score: {train_score}')
    print(f'Test Score: {test_score}')
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        print(f'AUC Score: {auc}')
    print()    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    viz = Visualizer(X=X_train, y=y_train, 
                    transformer=model.best_estimator_.named_steps.prep,
                    classifier=model.best_estimator_.named_steps.clf)
    
    viz.plot_confusion_matrix(y_test, y_pred)
    plt.show() # so it doesn't put them all at the end
    print()
    print()
    fitted_models[name] = {
        'auc_score': auc,
        'train_score': train_score,
        'test_score': test_score
    }
    if auc > best_auc_score:
        best_auc_score = auc
        best_model = model.best_estimator_
    
    print("*"*50)
    print('BEST MODEL SO FAR:', best_model)
    print()
    print()

# %%
model_output = pd.DataFrame(fitted_models).T
model_output

# %%
best_model

# %%
joblib.dump(best_model, '01_best_model')

# %%



