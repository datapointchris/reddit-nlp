"""
Preprocessors and estimators, along with their parameters for gridsearching.

Used with 'compare_models' function from the Reddit class, Model class maybe
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__max_features": [5000]
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
            "logreg__penalty": ["l1", "l2"],
            "logreg__C": [.01, .1, 1, 3]
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "abbr": "random_forest",
        "estimator": RandomForestClassifier(),
        "pipe_params":
            {
            "randomforest__n_estimators": [100, 200, 300],
            "randomforest__max_depth": [200],
            "randomforest__min_samples_leaf": [1, 2, 3],
            "randomforest__min_samples_split": [.0005, .001, .01]
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
            "svc__C": [1, 2, 3, 4, 5],
            "svc__kernel": ["linear", "poly", "rbf"],
            "svc__gamma": ["scale"],
            "svc__degree": [1, 2, 3, 4, 5],
            "svc__probability": [True]
        }
    }
}
