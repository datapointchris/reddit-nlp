import numpy as np
import requests
import json
import pandas as pd
from time import sleep
import datetime
import os
import argparse
import databases

headers = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
           'accept-encoding': 'gzip, deflate, sdch, br',
           'accept-language': 'en-GB,en;q=0.8,en-US;q=0.6,ml;q=0.4',
           'cache-control': 'max-age=0',
           'upgrade-insecure-requests': '1',
           'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}


class Scraper:

    def __init__(self):
        self.date = str(datetime.datetime.now().date())

    def scrape_subreddit(self, subreddit_list, sorting='new'):
        '''Scrapes a subreddit for post titles.

            subname: subreddit name to scrape

            pages: pages to scrape, typically 25 posts per page. Default is 42 for partial page scrape buffer.
            Reddit seems to have a soft limit of 1000 posts, can't seem to get around it.

            sorting: possible sort orders
                - new
                - rising
                - controversial
                - top
        '''
        self.sorting = sorting

        df = pd.DataFrame()

        for sub in subreddit_list:

            url = f'https://old.reddit.com/r/{sub}/{sorting}.json'
            post_titles = []
            prev_post_len = 0
            curr_post_len = 0
            after = None
            print(f'Scraping subreddit "{sub}"')

            while (prev_post_len == 0) or (prev_post_len != curr_post_len):
                prev_post_len = curr_post_len
                if after is None:
                    params = {}
                else:
                    params = {'after': after}
                response = requests.get(url, params=params, headers=headers)

                if response.status_code != 200:
                    print('Error:', response.status_code)
                    break

                post_json = response.json()
                for post in post_json['data']['children']:
                    title = post['data']['title']
                    if title not in post_titles:
                        post_titles.append(title)
                curr_post_len = len(post_titles)

                after = post_json['data']['after']
                sleep(.5)

            print(f'Success. {len(post_titles)} total posts for "{sub}"')

            data = pd.DataFrame(
                data={'title': post_titles, 'subreddit': sub, 'date': self.date})
            df = pd.concat([df, data], ignore_index=True)

        return df

    def save_to_csv(self, df):

        date = str(datetime.datetime.now().date())

        if not os.path.exists('../scraped_subreddits/'):
            os.mkdir('../scraped_subreddits/')

        for sub in df.subreddit.unique():
            mask = df['subreddit'] == sub
            sub_df = df[mask]
            df.to_csv(f'../scraped_subreddits/{sub}_{self.sorting}_{self.date}.csv', index=False)
            print(f'Saved "{sub}" to CSV')

    def save_to_sqlite(self, df):
        db = databases.Sqlite()
        connection = db.create_connection('reddit.sqlite')

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

    def save_choice(self, choice):
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

    scrape = Scraper()
    df = scrape.scrape_subreddit(subreddit_list, sorting=sorting)
    scrape.save_choice(save_location)
