import logging
from glob import glob

import pandas as pd

from util import databases
from util import scraper
import CONFIG

logger = logging.getLogger(__name__)


def data_selector(class_labels, data_source):
    '''Finds data for subreddits from selected data_source.

    data_source: scrape
        Scrape each subreddit in the list now.
        Returns DataFrame

    data_source: csv
        CSVs for subreddits in the 'scraped_subreddits' directory
        Prints subreddits with no CSV files
        Returns DataFrame

    data_source: mongo
        Creates connection to Mongo DB
        Queries DB for subreddits
        Returns DataFrame

    data_source: sqlite3
        Creates connection to SQLite DB
        Queries DB for subreddits
        Returns DataFrame

    data_source: postgres
        Creates connection to Postgres DB
        Queries DB for subreddits
        Returns DataFrame

    data_source: mysql
        Creates connection to Mysql DB
        Queries DB for subreddits
        Returns DataFrame

    Returns:
    df - DataFrame of selected data
    '''

    if data_source == 'scrape':
        scrape = scraper.Scraper()
        df = scrape.scrape_subreddit(class_labels)
        return df

    if data_source == 'csv':
        df = pd.DataFrame()
        for label in class_labels:
            csv_files = sorted(glob(f'{CONFIG.SCRAPED_SUBREDDITS_DIR}/*{label}*.csv'))
            if len(csv_files) < 1:
                raise ValueError(f'No data for "{label}" in "{data_source}" data_source')
            for csv_file in csv_files:
                data = pd.read_csv(csv_file)
                df = pd.concat([df, data], ignore_index=True)
        return df

    # === NOTE === # this bypasses the 'execute_read_query' function in the databases module...
    if data_source == 'sqlite':
        db = databases.Sqlite()
        connection = db.create_connection(CONFIG.DATA_DIR / 'reddit.sqlite')

        placeholders = ','.join('?' for label in class_labels)

        subreddit_query = f"""
        SELECT
            title,
            subreddit,
            date
        FROM subreddits
        WHERE subreddit IN (%s);""" % str(placeholders)

        cursor = connection.cursor()
        cursor.execute(subreddit_query, class_labels)

        column_names = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data=data, columns=column_names)
        df = df.drop_duplicates(subset='title')
        for label in class_labels:
            if len(df[df['subreddit'] == label]) == 0:
                raise ValueError(f'No data for "{label}" in "{data_source}" data_source')
        return df

    if data_source == 'mongo':
        db = databases.Mongo()
        return db.create_connection()

    if data_source == 'postgres':
        db = databases.Postgres()
        return db.create_connection()

    if data_source == 'mysql':
        db = databases.Mysql()
        return db.create_connection()
