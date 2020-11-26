
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
        'estimator': XGBClassifier(n_estimators=200),
        'pipe_params': {
            "xgbclassifier__max_depth": [5, 10]
        }
    },
    "logisticregression": {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(max_iter=1000, fit_intercept=False, C=.99),
        "pipe_params": {
            "logisticregression__solver": ["lbfgs", "saga"]
        }
    },
    "randomforestclassifier": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(min_samples_leaf=2, min_samples_split=.01),
        "pipe_params": {
            "randomforestclassifier__n_estimators": [300, 500, 1000],
            "randomforestclassifier__max_depth": np.linspace(400, 1000, 5, dtype=int)
        }
    },
  
    "multinomialnb": {
        "name": "Multinomial Bayes Classifier",
        "estimator": MultinomialNB(alpha=.1189),
        "pipe_params": {
            "multinomialnb__fit_prior": [True, False]
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "estimator": SVC(kernel="sigmoid"),
        "pipe_params": {
            "svc__C": [.99, 1]
        }
    },
    "sgdclassifier": {
        "name": "Stochastic Gradient Descent Classifier",
        "estimator": SGDClassifier(alpha=.0001, fit_intercept=True, penalty="l2"),
        "pipe_params":
            {
        }
    }
}


def plot_confusion_matrix(model, y_true, y_pred, classes, cmap='Blues'):
    '''
    Plots confusion matrix for fitted model, better than scikit-learn version
    '''
    cm = confusion_matrix(y_true, y_pred)
    fontdict = {'fontsize': 16}
    fig, ax = plt.subplots(figsize=(2.2 * len(classes), 2.2 * len(classes)))

    sns.heatmap(cm,
                annot=True,
                annot_kws=fontdict,
                fmt="d",
                square=True,
                cbar=False,
                cmap=cmap,
                ax=ax,
                norm=LogNorm(),  # to get color diff on small values
                vmin=0.00001  # to avoid non-positive error for '0' cells
                )

    ax.set_xlabel('Predicted labels', fontdict=fontdict)
    ax.set_ylabel('True labels', fontdict=fontdict)
    ax.set_yticklabels(
        labels=classes, rotation='horizontal', fontdict=fontdict)
    ax.set_xticklabels(labels=classes, rotation=20, fontdict=fontdict)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(5, shuffle=True, random_state=0)
GridSearch(model, X, y, cv=skf)