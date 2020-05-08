"""
Preprocessors and estimators, along with their parameters for gridsearching.
Used with "compare_models" function from the Reddit class, Model class maybe
"""

import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                             CountVectorizer, TfidfVectorizer)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, SGDClassifier)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ========================= CLASS LABELS ========================= #

class_labels_all = ["deeplearning", "tensorflow", "scikit_learn", "pandas", "bigdata", "aws",
                    "awscertifications", "css", "html", "javascript", "shittyprogramming",
                    "java", "sql", "learnsql", "postgresql", "softwarearchitecture", "scala",
                    "apachespark", "mongodb", "linux", "linux4noobs", "datascience", "machinelearning",
                    "etl", "python", "dataengineering"]

class_labels = ["scikit_learn", "pandas", "bigdata", "aws",
                "shittyprogramming", "java", "sql", "learnsql",
                "postgresql", "etl", "python", "dataengineering"]

class_labels_random = np.random.choice(class_labels_all, 8, replace=False)


# ========================= STOP WORDS ========================= #

useless_words = set(['postgres', 'big', 'panda', 'using', 'scikit', 'sklearn', 'apache', 'spark', 'lambda', 's3',
                     'does', 'looking', 'help', 'new', 'data', 'science', 'scientist', 'machine', 'learning', 'use',
                     'need', 'engineer', 'engineering'])

custom_stop_words = ENGLISH_STOP_WORDS.union(useless_words)


# ========================= PREPROCESSORS ========================= #

preprocessors = {
    "countvectorizer": {
        "name": "CountVectorizer",
        "preprocessor": CountVectorizer(),
        "pipe_params": {
            "countvectorizer__max_features": [5000],
            # "countvectorizer__max_df": [.3, .4, .5],
            "countvectorizer__ngram_range": [(1, 2)],
            "countvectorizer__stop_words": [custom_stop_words],
            # "countvectorizer__min_df": [4, 5, 6]
        }
    },
    "tfidfvectorizer": {
        "name": "TfidVectorizer",
        "preprocessor": TfidfVectorizer(),
        "pipe_params": {
            "tfidfvectorizer__strip_accents": [None],
            "tfidfvectorizer__stop_words": [custom_stop_words],
            "tfidfvectorizer__ngram_range": [(1, 2)],
            "tfidfvectorizer__max_features": [5000],
            "tfidfvectorizer__norm": ("l1", "l2"),
        }
    }
}


# ========================= ESTIMATORS ========================= #

estimators = {
    'xgbclassifier': {
        'name': 'XGBoost Classifier',
        'estimator': XGBClassifier(),
        'pipe_params': {
            "xgbclassifier__hidden_layer_sizes": [10, 25, 50],
            "xgbclassifier__n_estimators": [50, 100, 200],
            "xgbclassifier__max_depth": [5, 10, 20]
        }
    },
    'mlpclassifier': {
        'name': 'Multi Layer Percetpron Classifier',
        'estimator': MLPClassifier(),
        'pipe_params': {
            "mlpclassifier__hidden_layer_sizes": [50, 100, 200]
        }
    },
    "logisticregression": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(),
        "pipe_params": {
            "logisticregression__penalty": ["l2"],
            "logisticregression__C": [.01, .1, 1, 3],
            "logisticregression__max_iter": [1000],
            "logisticregression__solver": ["lbfgs", "saga"]
        }
    },
    "randomforestclassifier": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(),
        "pipe_params": {
            "randomforestclassifier__n_estimators": [100, 300],
            "randomforestclassifier__max_depth": np.linspace(5, 500, 5),
            "randomforestclassifier__min_samples_leaf": [1, 2, 3],
            "randomforestclassifier__min_samples_split": [.01, .05, .1]
        }
    },
    "kneighborsclassifier": {
        "name": "K Nearest Neighbors",
        "estimator": KNeighborsClassifier(),
        "pipe_params": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7],
            "kneighborsclassifier__metric": ["manhattan"]
        }
    },
    "multinomialnb": {
        "name": "Multinomial Bayes Classifier",
        "estimator": MultinomialNB(),
        "pipe_params": {
            "multinomialnb__fit_prior": [False],
            "multinomialnb__alpha": [.01, .1, 1]
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "estimator": SVC(),
        "pipe_params": {
            "svc__C": [1, 10, 100],
            "svc__kernel": ["rbf", "sigmoid", "poly"],
            "svc__gamma": ["scale"],
            "svc__probability": [False]
        }
    },
    "adaboostclassifier": {
        "name": "AdaBoost Classifier Logistic Regression",
        "estimator": AdaBoostClassifier(),
        "pipe_params": {
            "adaboostclassifier__learning_rate": [.001, .01, .1],
            "adaboostclassifier__n_estimators": [1, 2, 3]
        }
    },
    "baggingclassifierlog": {
        "name": "Bagging Classifier Logistic Regression",
        "estimator": BaggingClassifier(LogisticRegression(max_iter=1000)),
        "pipe_params": {
            "baggingclassifier__n_estimators": [5, 10, 20]
        }
    },
    "baggingclassifiermnb": {
        "name": "Bagging Classifier MultinomalNB",
        "estimator": BaggingClassifier(MultinomialNB()),
        "pipe_params": {
            "baggingclassifier__n_estimators": [10, 50, 100, 200, 500]
        }
    },
    "extratreesclassifier": {
        "name": "Extra Trees Classifier",
        "estimator": ExtraTreesClassifier(),
        "pipe_params": {
            "extratreesclassifier__bootstrap": [True],
            "extratreesclassifier__class_weight": [None],
            "extratreesclassifier__max_depth": [None],
            "extratreesclassifier__max_leaf_nodes": [None],
            "extratreesclassifier__min_samples_leaf": [1],
            "extratreesclassifier__min_samples_split": [2],
            "extratreesclassifier__min_weight_fraction_leaf": [0.0],
            "extratreesclassifier__n_estimators": [100, 300, 500],
        }
    },
    "gradientboostingclassifier": {
        "name": "Gradient Boosting Classifier",
        "estimator": GradientBoostingClassifier(),
        "pipe_params": {
            "gradientboostingclassifier__learning_rate": [0.1],
            "gradientboostingclassifier__max_depth": [3, 5],
            "gradientboostingclassifier__min_impurity_decrease": [0.0],
            "gradientboostingclassifier__min_samples_leaf": [1],
            "gradientboostingclassifier__min_samples_split": [2],
            "gradientboostingclassifier__min_weight_fraction_leaf": [0.0],
            "gradientboostingclassifier__n_estimators": [100, 300, 500]
        }
    },
    "passiveaggressiveclassifier": {
        "name": "Passive Agressive Classifier",
        "estimator": PassiveAggressiveClassifier(),
        "pipe_params":
            {
            "passiveaggressiveclassifier__C": [1.0],
            "passiveaggressiveclassifier__average": [False],
            "passiveaggressiveclassifier__class_weight": [None],
            "passiveaggressiveclassifier__early_stopping": [False],
            "passiveaggressiveclassifier__fit_intercept": [True],
            "passiveaggressiveclassifier__max_iter": [1000],
            "passiveaggressiveclassifier__n_iter_no_change": [5]
        }
    },
    "sgdclassifier": {
        "name": "Stochastic Gradient Descent Classifier",
        "estimator": SGDClassifier(),
        "pipe_params":
            {
            "sgdclassifier__alpha": [0.0001],
            "sgdclassifier__average": [False],
            "sgdclassifier__class_weight": [None],
            "sgdclassifier__early_stopping": [False],
            "sgdclassifier__epsilon": [0.1],
            "sgdclassifier__eta0": [0.0],
            "sgdclassifier__fit_intercept": [True],
            "sgdclassifier__l1_ratio": [0.15],
            "sgdclassifier__max_iter": [1000],
            "sgdclassifier__n_iter_no_change": [5],
            "sgdclassifier__n_jobs": [None],
            "sgdclassifier__penalty": ["l2", "l1", "elasticnet"],
            "sgdclassifier__power_t": [0.5]
        }
    },
    "nusvc": {
        "name": "Nu Support Vector Classifier",
        "estimator": NuSVC(),
        "pipe_params":
            {
            "nusvc__cache_size": [200, 400, 800],
            "nusvc__decision_function_shape": ["ovr"],
            "nusvc__degree": [3]
        }
    }
}
