"""
Preprocessors and estimators, along with their parameters for gridsearching.

Used with 'compare_models' function from the Reddit class, Model class maybe
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
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
    "lr": {
        "name": "Logistic Regression",
        "abbr": "lr",
        "estimator": LogisticRegression(),
        "pipe_params":
            {
            "lr__penalty": ["l1", "l2"],
            "lr__C": [.01, .1, 1, 3]
        }
    },
    "rf": {
        "name": "Random Forest",
        "abbr": "rf",
        "estimator": RandomForestClassifier(),
        "pipe_params":
            {
            "rf__n_estimators": [100, 200, 300],
            "rf__max_depth": [200],
            "rf__min_samples_leaf": [1, 2, 3],
            "rf__min_samples_split": [.0005, .001, .01]
        }
    },
    "knn": {
        "name": "K Nearest Neighbors",
        "abbr": "knn",
        "estimator": KNeighborsClassifier(),
        "pipe_params":
            {
            "knn__n_neighbors": [3, 5, 7],
            "knn__metric": ["manhattan"]
        }
    },
    "mnb": {
        "name": "Multinomial Bayes Classifier",
        "abbr": "mnb",
        "estimator": MultinomialNB(),
        "pipe_params":
            {
            "mnb__fit_prior": [False],
            "mnb__alpha": [0, .1, 1]
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
