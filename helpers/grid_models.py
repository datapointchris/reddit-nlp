"""
Preprocessors and estimators, along with their parameters for gridsearching.
Used with "compare_models" function from the Reddit class, Model class maybe
"""

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import (ElasticNet, LogisticRegression,
                                  PassiveAggressiveClassifier, SGDClassifier)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, NuSVC

preprocessors = {
    "count_vec": {
        "name": "CountVectorizer",
        "abbr": "count_vec",
        "processor": CountVectorizer(),
        "pipe_params": {
                "count_vec__max_features": [5000],
                "count_vec__max_df": [.3, .4, .5],
                "count_vec__ngram_range": [(1, 2)],
                "count_vec__stop_words": ["english"],
                "count_vec__min_df": [4, 5, 6]
        }
    },
    "tfidf": {
        "name": "TfidVectorizer",
        "abbr": "tfidf",
        "processor": TfidfVectorizer(),
        "pipe_params":
            {
                "tfidf__strip_accents": [None],
                "tfidf__stop_words": ["english"],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__max_features": [5000],
                "tfidf__use_idf": (True, False),
                "tfidf__norm": ("l1", "l2"),
        }
    }
}

estimators = {
    "logreg": {
        "name": "Logistic Regression",
        "abbr": "logreg",
        "estimator": LogisticRegression(),
        "pipe_params":
            {
            "logreg__penalty": ["l1", "l2", "elasticnet"],
            "logreg__C": [.01, .1, 1, 3],
            "logreg__max_iter": [200, 300, 500, 700],
            "logreg__solver": ["newton-cg", "lbfgs", "sag", "saga", "liblinear"]
        }
    },
    "randomforest": {
        "name": "Random Forest",
        "abbr": "randomforest",
        "estimator": RandomForestClassifier(),
        "pipe_params":
            {
            "randomforest__n_estimators": [200, 300],
            "randomforest__max_depth": [200],
            "randomforest__min_samples_leaf": [1, 2, 3],
            "randomforest__min_samples_split": [.001, .01]
        }
    },
    "knearest": {
        "name": "K Nearest Neighbors",
        "abbr": "knearest",
        "estimator": KNeighborsClassifier(),
        "pipe_params":
            {
            "knearest__n_neighbors": [3, 5, 7],
            "knearest__metric": ["manhattan"]
        }
    },
    "multinomialnb": {
        "name": "Multinomial Bayes Classifier",
        "abbr": "multinomialnb",
        "estimator": MultinomialNB(),
        "pipe_params":
            {
            "multinomialnb__fit_prior": [False],
            "multinomialnb__alpha": [.01, .1, 1]
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "abbr": "svc",
        "estimator": SVC(),
        "pipe_params":
            {
            "svc__C": [3, 4],
            "svc__kernel": ["rbf"],
            "svc__gamma": ["scale"],
            "svc__degree": [1],
            "svc__probability": [True]
        }
    },
    "ada": {
        "name": "AdaBoost Classifier",
        "abbr": "ada",
        "estimator": AdaBoostClassifier(),
        "pipe_params":
            {
            "ada__algorithm": [3, 4],
            "ada__base_estimator": ["rbf"],
            "ada__learning_rate": ["scale"],
            "ada__n_estimators": [1]
        }
    },
    "bag": {
        "name": "Bagging Classifier",
        "abbr": "bag",
        "estimator": BaggingClassifier(),
        "pipe_params":
            {"base_estimator": [None],
             "bootstrap": [True],
             "bootstrap_features": [False],
             "max_features": [1.0],
             "max_samples": [1.0],
             "n_estimators": [10],
             "n_jobs": [None],
             "oob_score": [False],
             "random_state": [None],
             "verbose": [0],
             "warm_start": [False]
             }
    },
    "extratrees": {
        "name": "Extra Trees Classifier",
        "abbr": "extratrees",
        "estimator": ExtraTreesClassifier(),
        "pipe_params":
            {"bootstrap": [False],
                "ccp_alpha": [0.0],
                "class_weight": [None],
                "criterion": ["gini"],
                "max_depth": [None],
                "max_features": ["auto"],
                "max_leaf_nodes": [None],
                "max_samples": [None],
                "min_impurity_decrease": [0.0],
                "min_impurity_split": [None],
                "min_samples_leaf": [1],
                "min_samples_split": [2],
                "min_weight_fraction_leaf": [0.0],
                "n_estimators": [100],
                "n_jobs": [None],
                "oob_score": [False],
                "random_state": [None],
                "verbose": [0],
                "warm_start": [False]
             }
    },
    "gradboost": {
        "name": "Gradient Boosting Classifier",
        "abbr": "gradboost",
        "estimator": GradientBoostingClassifier(),
        "pipe_params":
            {"ccp_alpha": [0.0],
             "criterion": ["friedman_mse"],
             "init": [None],
             "learning_rate": [0.1],
             "loss": ["deviance"],
             "max_depth": [3],
             "max_features": [None],
             "max_leaf_nodes": [None],
             "min_impurity_decrease": [0.0],
             "min_impurity_split": [None],
             "min_samples_leaf": [1],
             "min_samples_split": [2],
             "min_weight_fraction_leaf": [0.0],
             "n_estimators": [100],
             "n_iter_no_change": [None],
             "presort": ["deprecated"],
             "random_state": [None],
             "subsample": [1.0],
             "tol": [0.0001],
             "validation_fraction": [0.1],
             "verbose": [0],
             "warm_start": [False]
             }

    },
    "elastic": {
        "name": "ElasticNet Classifier",
        "abbr": "elastic",
        "estimator": ElasticNet(),
        "pipe_params":
            {"alpha": [1.0],
             "copy_X": [True],
             "fit_intercept": [True],
             "l1_ratio": [0.5],
             "max_iter": [1000],
             "normalize": [False],
             "positive": [False],
             "precompute": [False],
             "random_state": [None],
             "selection": ["cyclic"],
             "tol": [0.0001],
             "warm_start": [False]
             }
    },
    "passive": {
        "name": "Passive Agressive Classifier",
        "abbr": "passive",
        "estimator": PassiveAggressiveClassifier(),
        "pipe_params":
            {"C": [1.0],
                "average": [False],
                "class_weight": [None],
                "early_stopping": [False],
                "fit_intercept": [True],
                "loss": ["hinge"],
                "max_iter": [1000],
                "n_iter_no_change": [5],
                "n_jobs": [None],
                "random_state": [None],
                "shuffle": [True],
                "tol": [0.001],
                "validation_fraction": [0.1],
                "verbose": [0],
                "warm_start": [False]
             }
    },
    "sgd": {
        "name": "Stochastic Gradient Descent Classifier",
        "abbr": "sgd",
        "estimator": SGDClassifier(),
        "pipe_params":
            {"alpha": [0.0001],
                "average": [False],
                "class_weight": [None],
                "early_stopping": [False],
                "epsilon": [0.1],
                "eta0": [0.0],
                "fit_intercept": [True],
                "l1_ratio": [0.15],
                "learning_rate": ["optimal"],
                "loss": ["hinge"],
                "max_iter": [1000],
                "n_iter_no_change": [5],
                "n_jobs": [None],
                "penalty": ["l2"],
                "power_t": [0.5],
                "random_state": [None],
                "shuffle": [True],
                "tol": [0.001],
                "validation_fraction": [0.1],
                "verbose": [0],
                "warm_start": [False]
             }
    },
    "radneighbor": {
        "name": "Radius Neighbors Classifier",
        "abbr": "radneighbor",
        "estimator": RadiusNeighborsClassifier(),
        "pipe_params":
            {"algorithm": ["auto"],
             "leaf_size": [30],
             "metric": ["minkowski"],
             "metric_params": [None],
             "n_jobs": [None],
             "outlier_label": [None],
             "p": [2],
             "radius": [1.0],
             "weights": ["uniform"]}
    },
    "nusvc": {
        "name": "Nu Support Vector Classifier",
        "abbr": "nusvc",
        "estimator": NuSVC(),
        "pipe_params":
            {"break_ties": [False],
             "cache_size": [200],
             "class_weight": [None],
             "coef0": [0.0],
             "decision_function_shape": ["ovr"],
             "degree": [3],
             "gamma": ["scale"],
             "kernel": ["rbf"],
             "max_iter": [-1],
             "nu": [0.5],
             "probability": [False],
             "random_state": [None],
             "shrinking": [True],
             "tol": [0.001],
             "verbose": [False]}
    }

}
