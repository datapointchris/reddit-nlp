import datetime
import time
from pathlib import Path
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (classification_report,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split,)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from helpers import custom_stop_words, load_sqlite, plot_confusion_matrix

# ============================================================== #
# ==================== VARIABLES AND MODELS ==================== #

date = str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M'))
now = datetime.datetime.now()


database = 'reddit.sqlite'
class_labels = ['python', 'javascript', 'html', 'dataengineering']
random_state = 77
cross_val_splits = 2
verbose = 1

save_directory = 'compare_models_output'
save_dir = Path(save_directory, str(now))
os.makedirs(save_dir)

preprocessors = {
    'tfidf': {
        'name': 'TF-IDF Vectorizer',
        'preprocessor': TfidfVectorizer(stop_words=custom_stop_words),
        'params': {
            "prep__ngram_range": [(1, 2)],
            "prep__max_df": [.9],
            "prep__use_idf": [True],
            "prep__norm": ["l2"]
        }
    }
}

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
        "estimator": SGDClassifier(alpha=.0001, fit_intercept=True, penalty="l2", loss='modified_huber'),
        "params":
            {
        }
    },
    'xgbclassifier': {
        'name': 'XGBoost Classifier',
        'estimator': XGBClassifier(),
        'params': {
            "clf__max_depth": [3, 5, 10],
            "clf__learning_rate": [.001, .01, .1],
            "clf__n_estimators": [100, 500, 1000],
            # "clf__objective": ['binary:logistic', 'multi:softprob'],
            # "clf__booster": ['gbtree', 'gblinear', 'dart'],
            # "clf__gamma": [0, 1, 5],
            # "clf__reg_lambda": [0, .5, 1],
        }
    }
}
# ============================================================== #
# ============================================================== #


def main():

    df = load_sqlite(database=database, class_labels=class_labels)

    X = df['title']
    y = df['subreddit']

    labeler = LabelEncoder()
    y = labeler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model_comparison_df = pd.DataFrame(columns=[
        'preprocessor',
        'estimator',
        'best_train_score',
        'best_test_score',
        'time_weighted_score',
        'roc_auc',
        'train_test_variance',
        'fit_time_seconds',
        'predict_time_seconds',
        'best_params',
        'subreddits',
        'date'
    ])

    for name, est in tqdm(estimators.items()):
        for prep in preprocessors.values():
            print("*" * 50)
            print(f"Fitting model with {prep.get('name')} and {est.get('name')}")
            print()
            try:
                pipe = Pipeline([
                    ('prep', prep.get('preprocessor')),
                    ('clf', est.get('estimator'))
                ])
                pipe_params = dict()
                pipe_params.update(prep.get('params'))
                pipe_params.update(est.get('params'))
                skf = StratifiedKFold(n_splits=cross_val_splits, shuffle=True, random_state=random_state)
                model = GridSearchCV(pipe, param_grid=pipe_params, cv=skf, verbose=verbose, n_jobs=-1)
                model.fit(X_train, y_train)
            except Exception as e:
                print(f'ERROR BUILDING AND TRAINING MODEL: {e}')
                continue

            train_score = model.score(X_train, y_train)
            predict_start_time = time.time()
            test_score = model.score(X_test, y_test)
            predict_elapsed_time = time.time() - predict_start_time
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                print(f'AUC Score: {roc_auc}')

            subreddits = (', ').join(labeler.classes_)
            time_weighted_score = test_score / (model.refit_time_ + predict_elapsed_time) * 1000
            train_test_score_variance = (train_score - test_score) / train_score

            print(f'Train Score: {train_score}')
            print(f'Test Score: {test_score}')
            print()

            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred, digits=3, target_names=class_labels))

            plot_confusion_matrix(model, y_test, y_pred, classes=labeler.classes_)
            plt.savefig(save_dir / f'{name}_confusion_matrix.png')
            print()
            print()

            # add the model result to the df
            model_comparison_df.loc[len(model_comparison_df)] = [
                prep.get('name'),
                est.get('name'),
                round(train_score, 3),
                round(test_score, 3),
                round(time_weighted_score, 3),
                round(roc_auc, 3) if roc_auc else 'na',
                round(train_test_score_variance, 3),
                round(model.refit_time_, 3),
                round(predict_elapsed_time, 3),
                model.best_params_,
                subreddits,
                now
            ]

    print('Saving comparison df to CSV')
    try:
        model_comparison_df.to_csv(save_dir / 'model_comparison.csv')
    except FileNotFoundError:
        print('ERROR SAVING MODEL:')
    except UnboundLocalError:
        print('No compare_df saved.  Error fitting models:')


# ========================= MAIN ========================= #

if __name__ == "__main__":
    main_start = time.time()
    main()
    print(f'Total Run Time: {round((time.time() - main_start) / 60, 2)} minutes')
