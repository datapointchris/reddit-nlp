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
    # "logreg": {
    #     "name": "Logistic Regression",
    #     "abbr": "logreg",
    #     "estimator": LogisticRegression(),
    #     "pipe_params":
    #         {
    #         "logreg__penalty": ["l1", "l2", "elasticnet"],
    #         "logreg__C": [.01, .1, 1, 3],
    #         "logreg__max_iter": [200, 300, 500, 700],
    #         "logreg__solver": ["newton-cg", "lbfgs", "sag", "saga", "liblinear"]
    #     }
    # },
    # "randomforest": {
    #     "name": "Random Forest",
    #     "abbr": "randomforest",
    #     "estimator": RandomForestClassifier(),
    #     "pipe_params":
    #         {
    #         "randomforest__n_estimators": [200, 300],
    #         "randomforest__max_depth": [200],
    #         "randomforest__min_samples_leaf": [1, 2, 3],
    #         "randomforest__min_samples_split": [.001, .01]
    #     }
    # },
    # "knearest": {
    #     "name": "K Nearest Neighbors",
    #     "abbr": "knearest",
    #     "estimator": KNeighborsClassifier(),
    #     "pipe_params":
    #         {
    #         "knearest__n_neighbors": [3, 5, 7],
    #         "knearest__metric": ["manhattan"]
    #     }
    # },
    # "multinomialnb": {
    #     "name": "Multinomial Bayes Classifier",
    #     "abbr": "multinomialnb",
    #     "estimator": MultinomialNB(),
    #     "pipe_params":
    #         {
    #         "multinomialnb__fit_prior": [False],
    #         "multinomialnb__alpha": [.01, .1, 1]
    #     }
    # },
    # "svc": {
    #     "name": "Support Vector Classifier",
    #     "abbr": "svc",
    #     "estimator": SVC(),
    #     "pipe_params":
    #         {
    #         "svc__C": [3, 4],
    #         "svc__kernel": ["rbf"],
    #         "svc__gamma": ["scale"],
    #         "svc__degree": [1],
    #         "svc__probability": [True]
    #     }
    # },
    # "ada": {
    #     "name": "AdaBoost Classifier",
    #     "abbr": "ada",
    #     "estimator": AdaBoostClassifier(),
    #     "pipe_params":
    #         {
    #         "ada__learning_rate": [.001, .01, .1],
    #         "ada__n_estimators": [1,2,3]
    #     }
    # },
    # "bag": {
    #     "name": "Bagging Classifier",
    #     "abbr": "bag",
    #     "estimator": BaggingClassifier(),
    #     "pipe_params":
    #         {
    #          "bag__bootstrap": [True, False],
    #          "bag__bootstrap_features": [False, True],
    #          "bag__max_features": [1.0],
    #          "bag__max_samples": [1.0],
    #          "bag__n_estimators": [5,10,20]
    #          }
    # },
    # "extratrees": {
    #     "name": "Extra Trees Classifier",
    #     "abbr": "extratrees",
    #     "estimator": ExtraTreesClassifier(),
    #     "pipe_params":
    #         {
    #             "extratrees__bootstrap": [False],
    #             "extratrees__ccp_alpha": [0.0],
    #             "extratrees__class_weight": [None],
    #             "extratrees__max_depth": [None],
    #             "extratrees__max_features": [2000,5000],
    #             "extratrees__max_leaf_nodes": [None],
    #             "extratrees__max_samples": [None],
    #             "extratrees__min_impurity_decrease": [0.0],
    #             "extratrees__min_impurity_split": [None],
    #             "extratrees__min_samples_leaf": [1],
    #             "extratrees__min_samples_split": [2],
    #             "extratrees__min_weight_fraction_leaf": [0.0],
    #             "extratrees__n_estimators": [100],
    #          }
    # },
    # "gradboost": {
    #     "name": "Gradient Boosting Classifier",
    #     "abbr": "gradboost",
    #     "estimator": GradientBoostingClassifier(),
    #     "pipe_params":
    #         {"gradboost__ccp_alpha": [0.0],
    #          "gradboost__learning_rate": [0.1],
    #          "gradboost__max_depth": [3],
    #          "gradboost__max_features": [None],
    #          "gradboost__max_leaf_nodes": [None],
    #          "gradboost__min_impurity_decrease": [0.0],
    #          "gradboost__min_impurity_split": [None],
    #          "gradboost__min_samples_leaf": [1],
    #          "gradboost__min_samples_split": [2],
    #          "gradboost__min_weight_fraction_leaf": [0.0],
    #          "gradboost__n_estimators": [100]
    #          }

    # },
    # "elastic": {
    #     "name": "ElasticNet Classifier",
    #     "abbr": "elastic",
    #     "estimator": ElasticNet(),
    #     "pipe_params":
    #         {"elastic__alpha": [1.0],
    #          "elastic__copy_X": [True],
    #          "elastic__fit_intercept": [True],
    #          "elastic__l1_ratio": [0.5],
    #          "elastic__max_iter": [1000],
    #          "elastic__normalize": [False, True],
    #          }
    # },
    "passive": {
        "name": "Passive Agressive Classifier",
        "abbr": "passive",
        "estimator": PassiveAggressiveClassifier(),
        "pipe_params":
            {
                "passive__C": [1.0],
                "passive__average": [False],
                "passive__class_weight": [None],
                "passive__early_stopping": [False],
                "passive__fit_intercept": [True],
                "passive__max_iter": [1000],
                "passive__n_iter_no_change": [5]
             }
    },
    "sgd": {
        "name": "Stochastic Gradient Descent Classifier",
        "abbr": "sgd",
        "estimator": SGDClassifier(),
        "pipe_params":
            {
                "sgd__alpha": [0.0001],
                "sgd__average": [False],
                "sgd__class_weight": [None],
                "sgd__early_stopping": [False],
                "sgd__epsilon": [0.1],
                "sgd__eta0": [0.0],
                "sgd__fit_intercept": [True],
                "sgd__l1_ratio": [0.15],
                "sgd__max_iter": [1000],
                "sgd__n_iter_no_change": [5],
                "sgd__n_jobs": [None],
                "sgd__penalty": ["l2","l1","elasticnet"],
                "sgd__power_t": [0.5]
             }
    },
    "nusvc": {
        "name": "Nu Support Vector Classifier",
        "abbr": "nusvc",
        "estimator": NuSVC(),
        "pipe_params":
            {"nusvc__break_ties": [False],
             "nusvc__cache_size": [200],
             "nusvc__class_weight": [None],
             "nusvc__coef0": [0.0],
             "nusvc__decision_function_shape": ["ovr"],
             "nusvc__degree": [3]
}
    }

}
