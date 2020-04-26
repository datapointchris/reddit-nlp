import argparse
import datetime
import functools
import json
import logging
import logging.handlers
import os
import sys
import time

import pandas as pd
import requests
from tqdm import tqdm

# set path to current working directory for cron job
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# path hack, I know it's gross
sys.path.insert(0, os.path.abspath('..'))

from util import databases


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
file_handler = logging.handlers.RotatingFileHandler(filename='../logs/scraper.log', maxBytes=10000000, backupCount=10)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

headers = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
           'accept-encoding': 'gzip, deflate, sdch, br',
           'accept-language': 'en-GB,en;q=0.8,en-US;q=0.6,ml;q=0.4',
           'cache-control': 'max-age=0',
           'upgrade-insecure-requests': '1',
           'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}


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


class Scraper:

    def __init__(self):
        self.date = str(datetime.datetime.now().date())

    @function_timer
    def scrape_subreddit(self, subreddit_list, sorting='new'):
        '''Scrapes a subreddit for post titles.

            subreddit_list: list of subreddits to scrape

            sorting: possible sort orders
                - new
                - rising
                - controversial
                - top
        '''
        self.sorting = sorting

        df = pd.DataFrame()

        for i, sub in enumerate(tqdm(subreddit_list), start=1):
            url = f'https://old.reddit.com/r/{sub}/{sorting}.json'
            post_titles = []
            after = None
            logger.info(f'Scraping subreddit "{sub}", {i} of {len(subreddit_list)}')

            for _ in range(40):
                if after is None:
                    params = {}
                else:
                    params = {'after': after}
                try:
                    response = requests.get(
                        url, params=params, headers=headers, timeout=None)
                except(ConnectionError, ConnectionResetError) as e:
                    logger.exception(f'Error scraping subreddit {sub}: {e}')
                    continue
                if response.status_code != 200:
                    logger.info(f'Error: {response.status_code}')
                    continue

                post_json = response.json()
                for post in post_json['data']['children']:
                    title = post['data']['title']
                    if title not in post_titles:
                        post_titles.append(title)

                after = post_json['data']['after']
                time.sleep(.5)

            logger.info(f'{len(post_titles)} scraped for "{sub}"')

            data = pd.DataFrame(
                data={'title': post_titles, 'subreddit': sub, 'date': self.date})
            df = pd.concat([df, data], ignore_index=True)

        return df

    def save_to_csv(self, df):

        if not os.path.exists('../data/scraped_subreddits/'):
            os.mkdir('../data/scraped_subreddits/')

        for sub in df.subreddit.unique():
            mask = df['subreddit'] == sub
            sub_df = df[mask]
            sub_df.to_csv(
                f'../data/scraped_subreddits/{sub}_{self.sorting}_{self.date}.csv', index=False)
            print(f'Saved "{sub}" to CSV')

    def save_to_sqlite(self, df):
        db = databases.Sqlite()
        connection = db.create_connection('../data/reddit.sqlite')

        create_subreddits_table = """
        CREATE TABLE IF NOT EXISTS subreddits (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          subreddit TEXT NOT NULL,
          date TEXT NOT NULL
        );
        """

        db.execute_query(connection, create_subreddits_table)
        df.to_sql(name='subreddits', con=connection,
                  if_exists='append', index=False)
        print('Data saved to sqlite database successfully')

    def save_to_postgres(self, df):
        return 'save to postgres'
        # YEAH

    def save_to_mongo(self, df):
        return 'save to mongo'
        # definitely a FUTURE addition

    def save_to_mysql(self, df):
        return 'save to mysql'
        # for django or something

    def save_choice(self, df, choice):
        '''
        Choice to save the scraped dataframe.
        Choices:
        'csv', 'sqlite', 'postgres', 'mongo', 'mysql'
        '''
        save_options = {
            'csv': self.save_to_csv,
            'sqlite': self.save_to_sqlite,
            'postgres': self.save_to_postgres,
            'mongo': self.save_to_mongo,
            'mysql': self.save_to_mysql
        }
        save_function = save_options.get(choice, self.save_to_csv)
        return save_function(df)


def main():

    logger.info('PROGRAM STARTED')
    scrape = Scraper()
    logger.info('Scraping Subreddits')
    df = scrape.scrape_subreddit(subreddit_list, sorting=sorting)
    logger.info(f'Saving data to {save_location}')
    scrape.save_choice(df, save_location)
    logger.info('PROGRAM FINISHED')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Subreddit Scraper',
                                     description='''
                                     Choose subreddits to scrape, how to sort them, and how to save them.
                                     Sort order options: "new", "rising", "controversial", "top". Default is "new".
                                     Save option is "csv", "sqlite", "postgres", "mongo", "mysql". Default is "csv".''')

    parser.add_argument('--config', action='store',
                        help='Configuration file for scraper')
    parser.add_argument('-subs', '--subreddit_list', action='store', nargs='+',
                        help='Subreddits to scrape, no quotes, no brackets')
    parser.add_argument('--sorting', action='store', choices=['new', 'rising', 'controversial', 'top'],
                        default='new', help='Sort order for subreddits to scrape.')
    parser.add_argument('--save', action='store', choices=['csv', 'sqlite', 'postgres', 'mongo', 'mysql'],
                        default='csv', help='How/where to save scraped subreddit.')

    args = parser.parse_args()

    if args.config:
        config_file = json.load(open(args.config))
        subreddit_list = config_file['subreddit_list']
        sorting = config_file['sorting']
        save_location = config_file['save_location']
    else:
        subreddit_list = args.subreddit_list
        sorting = args.sorting
        save_location = args.save

    main()
