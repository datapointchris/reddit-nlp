"""
Preprocessors and estimators, along with their parameters for gridsearching.
Used with "compare_models" function from the Reddit class, Model class maybe
"""
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, SGDClassifier)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

# ========================= CLASS LABELS ========================= #

class_labels_all = ["deeplearning", "tensorflow", "scikit_learn", "bigdata", "aws",
                    "awscertifications", "css", "html", "javascript", "shittyprogramming",
                    "java", "sql", "learnsql", "postgresql", "softwarearchitecture", "scala",
                    "apachespark", "mongodb", "linux", "linux4noobs", "datascience", "machinelearning",
                    "etl", "python", "dataengineering"]


def get_random_class_labels(num=8):
    return np.random.choice(class_labels_all, num, replace=False)


class_labels_random = get_random_class_labels()

# ========================= STOP WORDS ========================= #

useless_words = set(['postgres', 'big', 'panda', 'using', 'scikit', 'sklearn', 'apache', 'spark', 'lambda', 's3',
                     'does', 'looking', 'help', 'new', 'data', 'science', 'scientist', 'machine', 'learning', 'use',
                     'need', 'engineer', 'engineering'])

custom_stop_words = ENGLISH_STOP_WORDS.union(useless_words).union(set(class_labels_all))


# ========================= PREPROCESSORS ========================= #

preprocessors = {
    "tfidfvectorizer": {
        "name": "TfidfVectorizer",
        "preprocessor": TfidfVectorizer(stop_words=custom_stop_words),
        "pipe_params": {
            "tfidfvectorizer__strip_accents": [None, 'ascii', 'unicode'],
            "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
            "tfidfvectorizer__max_features": [5000, 6000, 7000],
            "tfidfvectorizer__min_df": [1],
            "tfidfvectorizer__max_df": [.95001, .99001, 1],
            "tfidfvectorizer__norm": ["l2"],
            "tfidfvectorizer__use_idf": [True, False]
        }
    }
}

original_preprocessors = {
    "tfidfvectorizer": {
        "name": "TfidfVectorizer",
        "preprocessor": TfidfVectorizer(stop_words=custom_stop_words),
        "pipe_params": {
            "tfidfvectorizer__strip_accents": [None, 'ascii', 'unicode'],
            "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
            "tfidfvectorizer__max_features": [5000, 6000, 7000],
            "tfidfvectorizer__min_df": np.arange(2, 20, 4),
            "tfidfvectorizer__max_df": np.linspace(.8, .99, 5),
            "tfidfvectorizer__norm": ("l1", "l2"),
            "tfidfvectorizer__use_idf": [True, False]
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
    # TOO SLOW
    # 'mlpclassifier': {
    #     'name': 'MLPClassifier',
    #     'estimator': MLPClassifier(),
    #     'pipe_params': {
    #         "mlpclassifier__hidden_layer_sizes": [(100,), (250,), (500,)],
    #         "mlpclassifier__alpha": np.linspace(.0001, 1, 5),
    #         "mlpclassifier__activation": ['relu']
    #     }
    # },
    "logisticregression": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(max_iter=1000),
        "pipe_params": {
            "logisticregression__C": [.5, .745, .99],
            "logisticregression__solver": ["lbfgs", "saga"],
            "logisticregression__fit_intercept": [False]
        }
    },
    "randomforestclassifier": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(),
        "pipe_params": {
            "randomforestclassifier__n_estimators": [100, 300],
            "randomforestclassifier__max_depth": np.linspace(5, 500, 5, dtype=int),
            "randomforestclassifier__min_samples_leaf": [1, 2, 3],
            "randomforestclassifier__min_samples_split": [.01, .05, .1]
        }
    },
    "kneighborsclassifier": {
        "name": "K Nearest Neighbors",
        "estimator": KNeighborsClassifier(),
        "pipe_params": {
            "kneighborsclassifier__n_neighbors": [7, 10, 15],
            "kneighborsclassifier__metric": ["manhattan"]
        }
    },
    "multinomialnb": {
        "name": "Multinomial Bayes Classifier",
        "estimator": MultinomialNB(),
        "pipe_params": {
            "multinomialnb__fit_prior": [True, False],
            "multinomialnb__alpha": np.linspace(.01, .99, 10)
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "estimator": SVC(),
        "pipe_params": {
            "svc__C": np.linspace(.01, .99, 10),
            "svc__kernel": ["sigmoid"]
        }
    },
    "adaboostclassifier": {
        "name": "AdaBoost Classifier",
        "estimator": AdaBoostClassifier(),
        "pipe_params": {
            "adaboostclassifier__learning_rate": [.1],
            "adaboostclassifier__n_estimators": [200, 500]
        }
    },
    # TOO SLOW #
    # "baggingclassifierlog": {
    #     "name": "Bagging Classifier Logistic Regression",
    #     "estimator": BaggingClassifier(LogisticRegression(max_iter=1000)),
    #     "pipe_params": {
    #         "baggingclassifier__n_estimators": [100, 200]
    #     }
    # },
    "baggingclassifier": {
        "name": "Bagging Classifier",
        "estimator": BaggingClassifier(),
        "pipe_params": {
            "baggingclassifier__n_estimators": [50, 100, 200]
        }
    },
    "extratreesclassifier": {
        "name": "Extra Trees Classifier",
        "estimator": ExtraTreesClassifier(),
        "pipe_params": {
            "extratreesclassifier__bootstrap": [True],
            "extratreesclassifier__n_estimators": [300, 500, 700],
        }
    },
    # REQUIRES DENSE MATRIX INSTEAD OF SPARSE
    # "histgradientboostingclassifier": {
    #     "name": "Hist Gradient Boosting Classifier",
    #     "estimator": HistGradientBoostingClassifier(),
    #     "pipe_params": {
    #         "histgradientboostingclassifier__max_iter": [100, 300, 500],
    #         "histgradientboostingclassifier__l2_regularization": np.linspace(.1, .9, 10)
    #     }
    # },
    "passiveaggressiveclassifier": {
        "name": "Passive Agressive Classifier",
        "estimator": PassiveAggressiveClassifier(),
        "pipe_params":
            {
            "passiveaggressiveclassifier__C": np.linspace(0, 1, 20),
            "passiveaggressiveclassifier__fit_intercept": [True, False],
        }
    },
    "sgdclassifier": {
        "name": "Stochastic Gradient Descent Classifier",
        "estimator": SGDClassifier(),
        "pipe_params":
            {
            "sgdclassifier__alpha": [.0001],
            "sgdclassifier__fit_intercept": [True, False],
            "sgdclassifier__penalty": ["l2"],
        }
    },
    "linearsvc": {
        "name": "Linear SVC",
        "estimator": LinearSVC(),
        "pipe_params":
            {
            "linearsvc__C": np.linspace(.01, 1, 5),
            "linearsvc__fit_intercept": [True, False],
        }
    }
}

original_estimators = {
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
        'name': 'MLPClassifier',
        'estimator': MLPClassifier(),
        'pipe_params': {
            "mlpclassifier__hidden_layer_sizes": [(100,), (250,), (500,)],
            "mlpclassifier__alpha": np.linspace(.0001, 1, 5),
            "mlpclassifier__activation": ['logistic', 'relu']
        }
    },
    "logisticregression": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(max_iter=1000),
        "pipe_params": {
            "logisticregression__C": np.linspace(.01, .99, 5),
            "logisticregression__solver": ["lbfgs", "saga"],
            "logisticregression__fit_intercept": [True, False]
        }
    },
    "randomforestclassifier": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(),
        "pipe_params": {
            "randomforestclassifier__n_estimators": [100, 300],
            "randomforestclassifier__max_depth": np.linspace(5, 500, 5, dtype=int),
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
            "multinomialnb__fit_prior": [True, False],
            "multinomialnb__alpha": np.linspace(.01, .99, 10)
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "estimator": SVC(),
        "pipe_params": {
            "svc__C": np.linspace(.01, .99, 10),
            "svc__kernel": ["rbf", "sigmoid", "poly"]
        }
    },
    "adaboostclassifier": {
        "name": "AdaBoost Classifier",
        "estimator": AdaBoostClassifier(),
        "pipe_params": {
            "adaboostclassifier__learning_rate": [.001, .01, .1],
            "adaboostclassifier__n_estimators": [50, 100, 200]
        }
    },
    "baggingclassifierlog": {
        "name": "Bagging Classifier Logistic Regression",
        "estimator": BaggingClassifier(LogisticRegression(max_iter=1000)),
        "pipe_params": {
            "baggingclassifier__n_estimators": [50, 100, 200]
        }
    },
    "baggingclassifier": {
        "name": "Bagging Classifier",
        "estimator": BaggingClassifier(),
        "pipe_params": {
            "baggingclassifier__n_estimators": [50, 100, 200]
        }
    },
    "extratreesclassifier": {
        "name": "Extra Trees Classifier",
        "estimator": ExtraTreesClassifier(),
        "pipe_params": {
            "extratreesclassifier__bootstrap": [True, False],
            "extratreesclassifier__n_estimators": [100, 300, 500],
        }
    },
    # REQUIRES DENSE MATRIX INSTEAD OF SPARSE
    # "histgradientboostingclassifier": {
    #     "name": "Hist Gradient Boosting Classifier",
    #     "estimator": HistGradientBoostingClassifier(),
    #     "pipe_params": {
    #         "histgradientboostingclassifier__max_iter": [100, 300, 500],
    #         "histgradientboostingclassifier__l2_regularization": np.linspace(.1, .9, 10)
    #     }
    # },
    "passiveaggressiveclassifier": {
        "name": "Passive Agressive Classifier",
        "estimator": PassiveAggressiveClassifier(),
        "pipe_params":
            {
            "passiveaggressiveclassifier__C": np.linspace(0, 1, 20),
            "passiveaggressiveclassifier__fit_intercept": [True, False],
        }
    },
    "sgdclassifier": {
        "name": "Stochastic Gradient Descent Classifier",
        "estimator": SGDClassifier(),
        "pipe_params":
            {
            "sgdclassifier__alpha": np.linspace(.0001, .1, 5),
            "sgdclassifier__fit_intercept": [True, False],
            "sgdclassifier__l1_ratio": np.linspace(0, 1, 5),
            "sgdclassifier__penalty": ["l2", "l1", "elasticnet"],
        }
    },
    "linearsvc": {
        "name": "Linear SVC",
        "estimator": LinearSVC(),
        "pipe_params":
            {
            "linearsvc__C": np.linspace(.1, 10, 5),
            "linearsvc__fit_intercept": [True, False],
        }
    }
}