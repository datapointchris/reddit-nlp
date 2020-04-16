import datetime
import json
import time
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools
import requests
import seaborn as sns
from itertools import combinations
import wordcloud
from PIL import Image
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from grid_models import estimators, preprocessors

# use this class and functions the same way as StandardScaler is used.
# Use the class to find the functions
# call the functions to modify data assigned to a variable.

def function_timer(func):
    """Prints runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        try:
            logger.info(f'Elapsed time: {round(elapsed_time/60,2)} minutes for {repr(func.__name__)}')
        except:
            print(f"Elapsed time: {round(elapsed_time/60,2)} minutes for function: '{repr(func.__name__)}'")
        return value
    return wrapper


class Reddit:

    def __init__(self):
        pass


    def compare_models(self, X_train, X_test, y_train, y_test, preprocessors=preprocessors, estimators=estimators, subs=None, cv=5, verbose=1, n_jobs=-1):

        # Set up the dataframe
        model_comparison_df = pd.DataFrame(columns=[
                                                'date',
                                                'preprocessor',
                                                'estimator',
                                                'best_params', 
                                                'best_train_score',
                                                'best_test_score',
                                                'variance',
                                                'prep_code',
                                                'est_code',
                                                'sub_list'
                                                ])         
        # fit a model for each combo of preprocessor and estimator
        for est in estimators.values():
            for prep in preprocessors.values():
                print(f"Fitting model with {prep['name']} and {est['name']}")
                pipe = Pipeline([(prep['abbr'], prep['processor']), (est['abbr'], est['estimator'])])
                pipe_params = dict()
                pipe_params.update(prep['pipe_params'])
                pipe_params.update(est['pipe_params'])
                model = GridSearchCV(pipe, param_grid=pipe_params, cv=cv, verbose=verbose, n_jobs=n_jobs)
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                now = datetime.datetime.now()
                subreddits = (', ').join(subs) if subs is not None else 'na'

                # add the model result to the df
                model_comparison_df.loc[len(model_comparison_df)] = [
                                        now,
                                        prep['name'],
                                        est['name'], 
                                        model.best_params_, 
                                        train_score,
                                        test_score,
                                        (train_score - test_score) / train_score * 100,
                                        prep['abbr'],
                                        est['abbr'],
                                        subreddits
                                    ]

        return model_comparison_df
# ====================================================================================== #
# =================================== VISUALIZER ======================================= #
class Visualizer:
    """
    Functions to vizualize data, instantiate with pandas df with a text column, labels column, and trained model

    'model': optional, use this if you have a trained model, (or transformer like Tfidf) that you want to use,
    otherwise use 'vect' or 'tfidf' to use one of those.
    
    'transformer': required if model is specified, otherwise optional. Defaults to TfidfVectorizer
    The transformer used in the pipeline of the trained model.
    If you want to specify parameters for vect or tfidf, you must instantiate your own model with parameters.

    Ex:
    tfidf = TfidfVectorizer(max_features=5000)

    """

    def __init__(self, df, text_column, labels_column, model=None, transformer=None):
        self.df = df
        self.text_column = text_column
        self.labels_column = labels_column
        self.model = model
        self.transformer = transformer
        self.labels = self.df[self.labels_column].unique()

        if model is None:
            if transformer == 'vect':
                vect = CountVectorizer()
                features_data = vect.transform(self.df[self.text_column]).toarray()
            else:
                model.transformer = 'tfidf'
                tfidf = TfidfVectorizer()
                features_data = tfidf.transform(self.df[self.text_column]).toarray()
        
        # assume the model has a pipeline with named steps
        features_data = self.model.named_steps[transformer].transform(self.df[self.text_column]).toarray()
        features_columns = self.model.named_steps[transformer].get_feature_names()
        self.features_df = pd.DataFrame(data=features_data, columns=features_columns)

        self.pairs = list(combinations(self.labels, 2))

        


    def make_cloud(self, labels_column=None, height=300, width=800, max_words=100, split=None, stopwords=None, colormap='viridis', background_color='black'):
        '''
        Inputs:
        text_column: name of text column in dataframe
        labels_column: column that contains the labels, if split=True
        height: height of each wordcloud
        width: width of each wordcloud
        max_words: max words for each wordcloud
        split: if True, wordcloud for each subreddit
        labels: must provide list of labels if split=True, to generate a wordcloud for each label
        stopwords: usually these are the same stopwords used by the tranformer (CountVectorizer or Tfidf)
        colormap: any choice from matplotlib gallery.  Find them with plt.cm.datad
            'random': picks a random colormap for each cloud.
        '''

        colormaps = [m for m in plt.cm.datad if not m.endswith("_r")]
        wc = wordcloud.WordCloud(max_words=max_words,
                                width=width,
                                height=height,
                                background_color=background_color,
                                colormap=np.random.choice(colormaps) if colormap == 'random' else colormap,
                                stopwords=stopwords)
        if split:
            for label in self.labels:
                cloud = wc.generate(
                    self.df[self.df[self.labels_column] == label][self.text_column].str.cat())
                plt.figure(figsize=(width/100, height*len(self.labels)/100), dpi=100)
                plt.title(label.upper(), fontdict={'fontsize': 15})
                plt.axis("off")
                plt.imshow(cloud.to_image(), interpolation='bilinear')

        else:
            cloud = wc.generate(self.df[self.text_column].str.cat())
            return cloud.to_image()


    ### CHECK ### Test output of tfidf vs countvectorizer
    def plot_most_common(self, num_features=20, standardize=False, include_combined=False):
        '''
        Plots the most common features for each subreddit in the DataFrame

        Parameters:

        df: original DataFrame

        features_df: should be output from transformer on df

            Example:
            features_df = pd.DataFrame(
                                    data={transformer}.transform(X).toarray(),
                                    columns={transformer}.get_feature_names())

        num_features: number of most common features to plot for each subreddit

        standardize: put all of the plots on the same scale

        combined: include a plot of the most common features of all of the subreddits combined

        Returns:

        plots

        '''

        fig, ax = plt.subplots(ncols=1,
                            nrows=len(self.labels) + int(1 if include_combined else 0),
                            figsize=(15, num_features/1.3*len(self.labels)))

        for subplot_idx, label in enumerate(self.labels):
            label_features = self.features_df.loc[self.df[self.labels_column] == label]
            label_top_words = label_features.sum().sort_values(ascending=False).head(num_features)[::-1]
            label_top_words.plot(kind='barh', ax=ax[subplot_idx])
            ax[subplot_idx].set_title(f'{num_features} Most Common Words for {label.upper()}', fontsize=16)
            
            if standardize:
                max_occurence = self.features_df.sum().max()*1.02
                ax[subplot_idx].set_xlim(0, max_occurence)

        if include_combined:
            most_common = self.features_df.sum().sort_values(ascending=False).head(num_features)[::-1]
            most_common.plot(kind='barh', ax=ax[subplot_idx+1])
            ax[subplot_idx+1].set_title(f'{num_features} Most Common Words for ({", ".join(self.labels).upper()})')
            
            if standardize:
                ax[subplot_idx+1].set_xlim(0, max_occurence)
        
        plt.tight_layout(h_pad=7)

    def plot_most_common_pairs(self, num_features=20):
        '''
        Plots the most common features for each subreddit in the DataFrame
        
        Parameters:
        
        num_features: number of most common features to plot for each subreddit
        
        Returns:
        
        plots
        
        '''
        fig, ax = plt.subplots(ncols=2, 
                            nrows=len(self.pairs), 
                        figsize=(16, (num_features/1.5)*len(self.pairs)))

        for i, pair in enumerate(self.pairs):

            # features for each pair
            feats_0 = self.features_df.loc[(self.df[self.labels_column] == pair[0])]
            feats_1 = self.features_df.loc[(self.df[self.labels_column] == pair[1])]
            # combined
            common_feats = feats_0.append(feats_1)
            # this is the most common between the two
            most_common = common_feats.sum().sort_values(ascending=False).head(num_features)[::-1]
            # plot
            feats_0[most_common.index].sum().plot.barh(ax=ax[i, 0], color='navy')
            feats_1[most_common.index].sum().plot.barh(ax=ax[i, 1], color='orange')
            ax[i, 0].set_title(f'Top {num_features} - {pair} \nSub: {pair[0].upper()}', fontsize=16, wrap=True)
            ax[i, 1].set_title(f'Top {num_features} - {pair} \nSub: {pair[1].upper()}', fontsize=16, wrap=True)
            max_occurence = common_feats.sum().max()*1.02
            ax[i, 0].set_xlim(0,max_occurence)
            ax[i, 1].set_xlim(0,max_occurence)
        plt.tight_layout()

    def plot_most_common_bar(self, num_features=20, stacked=False):

        most_common = self.features_df.sum().sort_values(ascending=False).head(num_features)
        groups = self.features_df.groupby(self.df[self.labels_column]).sum()[most_common.index].T.head(num_features)

        fig, ax = plt.subplots(figsize=(20, 10))

        if stacked is False:
            groups.plot.bar(ax=ax, width=.8, fontsize=15)
        else:
          groups.plot(kind='bar', ax=ax, width=.35, fontsize=15, stacked=True, )        
        
        ax.set_title(f'{num_features} Most Common Words', fontsize=20)
        ax.set_ylabel('# of Occurences', fontsize=15)
        ax.legend(fontsize=15, fancybox=True, framealpha=1, shadow=True, borderpad=1)

