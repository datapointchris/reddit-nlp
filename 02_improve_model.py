# %% [markdown]
# # Improve Model
#

# %% [markdown]
# ---

# %%

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
from helpers import load_sqlite
from helpers import resample_data

# %%
random_state = 77

labels = ['python', 'javascript', 'html']

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
grid = [
    {
        'resample_ada': [True, False],
        'resample_smote': [True, False],
        'resample_data_method': ['max', 'min', 'average', None],
        'resample_data_distribution': ['concatenate', 'resample'],
    }
]
param_grid = ParameterGrid(grid)

# %%
models_dict = dict()
best_score = 0.0
best_model = None

# %%
for i, param in enumerate(param_grid):
    print('*' * 50)
    print('PARAMS:')
    print(param)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    X_train, y_train = resample_data(
        X_train,
        y_train,
        sample_method=param['resample_data_method'],
        distribution=param['resample_data_distribution'],
        random_state=random_state,
    )

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
    print('*' * 50)
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
