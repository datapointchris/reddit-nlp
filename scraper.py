import argparse
import configparser
import datetime
import os
import sqlite3
import tempfile
import time
from io import StringIO
from pathlib import PurePath

import boto3
import pandas as pd
import requests
from tqdm import tqdm

headers = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
           'accept-encoding': 'gzip, deflate, sdch, br',
           'accept-language': 'en-GB,en;q=0.8,en-US;q=0.6,ml;q=0.4',
           'cache-control': 'max-age=0',
           'upgrade-insecure-requests': '1',
           'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}


class RedditScraper:
    """Scrapes one or many subreddits, with or without comments

    Use config files in 'scraper_configs' to configure scraper options.
    Documentation of the options is in the config files.

    Parameters
    ----------
    sorting : string, default='new'
        Reddit sort order for posts
        Options: 'new', 'rising', 'controversial', 'top'

    include_comments : bool, default=True
        Whether to include comments for each post

    save_method : string, default='csv'
        Save location (and method)
    """

    def __init__(self, project_name=None, sorting='new',
                 include_comments=True, save_method='sqlite', save_location='folder'):
        self.project_name = project_name
        self.sorting = sorting
        self.include_comments = include_comments
        self.save_method = save_method
        self.save_location = save_location
        self.date = str(datetime.datetime.now().date())
        self.project_directory = PurePath(__file__).parent / 'scraped' / self.project_name
        # if not os.path.exists(self.project_directory):
        #     os.makedirs(self.project_directory)

    def get_post_urls(self, sub_url):
        """Get post URLs for a given subreddit"""
        after = None
        posts_url_list = []
        for _ in range(40):
            if after is None:
                params = {}
            else:
                params = {'after': after}
            try:
                response = requests.get(
                    sub_url, params=params, headers=headers, timeout=None)
            except(ConnectionError, ConnectionResetError) as e:
                print(f'Error scraping {sub_url}: {e}')
                continue
            if response.status_code != 200:
                print(f'Error: {response.status_code}')
                continue

            post_json = response.json()
            post_urls = []
            for post in post_json['data']['children']:
                perma = post['data']['permalink']
                post_url = f'https://www.reddit.com{perma}'
                if post_url not in post_urls:
                    post_urls.append(post_url)
            posts_url_list.extend(post_urls)
            after = post_json['data']['after']
            time.sleep(.1)
        return posts_url_list

    def jsonify_url(self, url):
        '''Get json from url'''
        url = f'{url[:-1]}.json'
        return url

    def get_post(self, post_json):
        '''Get post info for current url'''
        post_id = post_json[0]['data']['children'][0]['data']['id']
        post_title = post_json[0]['data']['children'][0]['data']['title']
        post_body = post_json[0]['data']['children'][0]['data']['selftext']
        upvotes = post_json[0]['data']['children'][0]['data']['ups']
        return (post_id, post_title, post_body, upvotes)

    def get_comments(self, parent, comments):
        '''Get comments for current url'''
        if comments is None:
            comments = []
        for comment in parent['data']['children']:
            comment_id = comment['data'].get('id')
            comment_body = comment['data'].get('body')
            upvotes = comment['data'].get('ups')
            comments.append((comment_id, comment_body, upvotes))
            replies = comment['data'].get('replies')
            if replies:
                self.get_comments(replies, comments)
        return comments

    def scrape_subreddit(self, subreddit):
        """Scrapes a subreddit for post titles.

        Parameters
        ----------
        subreddit: subreddit to scrape

        Returns
        -------
        tuple: (posts_df, comments_df)
        If 'include_comments' is set to False, comments_df will return None
        """
        # create URL from subreddit name
        sub_url = f'https://reddit.com/r/{subreddit}/{self.sorting}.json'
        # get list of post URLs
        posts_url_list = self.get_post_urls(sub_url)

        posts_df = pd.DataFrame()
        comments_df = pd.DataFrame()
        for post_url in tqdm(posts_url_list):
            post_url = self.jsonify_url(post_url)
            try:
                response = requests.get(
                    post_url, params=None, headers=headers, timeout=None)
            except(ConnectionError, ConnectionResetError) as e:
                print(f'Connection error scraping {sub_url}: {e}')
                continue
            except Exception as e:
                print('Scraper Exception:', e)
                print(f'Skipping this post: {sub_url}')
                continue
            if response.status_code != 200:
                print(f'Response Error: {response.status_code}')
                continue
            post_json = response.json()
            post_id, post_title, post_body, upvotes = self.get_post(post_json)
            post_data = pd.DataFrame.from_records(
                [{'post_id': post_id,
                  'post_title': post_title,
                  'post_body': post_body,
                  'upvotes': upvotes,
                  'subreddit': subreddit,
                  'date': self.date
                  }])
            posts_df = pd.concat([posts_df, post_data], ignore_index=True)

            if self.include_comments:
                comments = self.get_comments(parent=post_json[1], comments=None)
                comments_data = pd.DataFrame.from_records(
                    [{'comment_id': comment[0],
                      'post_id': post_id,
                      'comment': comment[1],
                      'upvotes': comment[2]} for comment in comments])
                comments_df = pd.concat(
                    [comments_df, comments_data], ignore_index=True)

        print(f'{len(posts_df)} scraped for "{subreddit}"')
        if self.include_comments:
            return (posts_df, comments_df)
        return (posts_df, None)

    def create_sqlite_tables(self, df, table_name, subreddit, cursor):
        if table_name == 'posts':
            create_posts_table = """
                CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                post_title TEXT NOT NULL,
                post_body TEXT NOT NULL,
                upvotes INTEGER NOT NULL,
                subreddit TEXT NOT NULL,
                date TEXT NOT NULL
                );
                """
            cursor.execute(create_posts_table)
            for row in df.itertuples():
                row_values = [row.post_id, row.post_title, row.post_body, row.upvotes, row.subreddit, row.date]
                cursor.execute('''INSERT or REPLACE into posts (post_id, post_title, post_body, upvotes, subreddit, date)
                        values (?, ?, ?, ?, ?, ?)''', row_values)

        if table_name == 'comments':
            create_comments_table = """
                CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                comment TEXT,
                upvotes INTEGER
                );
                """
            cursor.execute(create_comments_table)
            for row in df.itertuples():
                row_values = [row.comment_id, row.post_id, row.comment, row.upvotes]
                cursor.execute('''INSERT or REPLACE into comments (comment_id, post_id, comment, upvotes)
                    values (?, ?, ?, ?)''', row_values)

        print('Data saved to sqlite database successfully')

    def save_to_csv(self, df, table_name, subreddit):
        filename = f'{self.date}_{subreddit}_{table_name}.csv'
        if self.save_location == 'folder':
            path = self.project_directory / 'csv'
            if not os.path.exists(path):
                os.makedirs(path)
            df.to_csv(f'{path}/{filename}', index=False)
        elif self.save_location == 's3':
            s3_path = f'{self.project_name}/csv/{filename}'
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            s3_resource = boto3.resource('s3')
            db_object = s3_resource.Object(bucket_name=s3_bucket_name, key=s3_path)
            db_object.put(Body=buffer.getvalue())

    def save_to_sqlite(self, df, table_name, subreddit):
        if self.save_location == 'folder':
            filename = f'{self.project_name}.sqlite'
            path = self.project_directory / 'sqlite'
            if not os.path.exists(path):
                os.makedirs(path)
            connect_path = f'{path}/{filename}'
            with sqlite3.connect(connect_path) as conn:
                cursor = conn.cursor()
                self.create_sqlite_tables(df=df, table_name=table_name,
                                          subreddit=subreddit, cursor=cursor)
        elif self.save_location == 's3':
            filename = f'{self.date}_{subreddit}.sqlite'
            s3_path = f'{self.project_name}/sqlite/{filename}'
            connect_path = tempfile.gettempdir() + '/temp.db'
            with sqlite3.connect(connect_path) as conn:
                cursor = conn.cursor()
                self.create_sqlite_tables(df=df, table_name=table_name,
                                          subreddit=subreddit, cursor=cursor)
                s3_resource = boto3.resource('s3')
                db_object = s3_resource.Object(bucket_name=s3_bucket_name, key=s3_path)
                db_object.put(Body=connect_path)

    def save_choice(self, df, choice, table_name, subreddit):
        '''
        Choice to save the scraped dataframe.
        Choices:
        'csv', 'sqlite', 'postgres', 'mongo', 'mysql', 's3'
        '''
        save_options = {
            'csv': self.save_to_csv,
            'sqlite': self.save_to_sqlite
        }
        save_function = save_options.get(choice, self.save_to_csv)
        return save_function(df, table_name, subreddit)


def main():
    print('PROGRAM STARTED')
    scraper = RedditScraper(project_name=project_name, sorting=sorting, include_comments=include_comments,
                            save_method=save_method, save_location=save_location)
    if len(subreddit_list) > 1:
        print(f'{len(subreddit_list)} total subreddits to scrape.')
    for i, sub in enumerate(tqdm(subreddit_list), start=1):
        print(f'Scraping "{sub}" subreddit')
        posts_df, comments_df = scraper.scrape_subreddit(sub)
        print(f'Saving {sub} posts data to {save_method}')
        scraper.save_choice(df=posts_df, choice=save_method, table_name='posts', subreddit=sub)
        if include_comments:
            print(f'Saving {sub} comments data to {save_method}')
            scraper.save_choice(df=comments_df, choice=save_method, table_name='comments', subreddit=sub)
    print('PROGRAM FINISHED')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Subreddit Scraper',
                                     description='''
                                     This program scrapes any number of subreddits with or without comments.
                                     Choose subreddits to scrape, how to sort them, and how/where to save them.
                                     Use config files in this directory to configure scraper options.
                                     Documentation of the options is in the config files.
                                     Usage:
                                     > python RedditScraper.py --config example.ini''')

    parser.add_argument('--config', action='store',
                        help='Configuration file for scraper.')

    args = parser.parse_args()

    scraper_config = configparser.ConfigParser()
    scraper_config.read(args.config)

    project_name = scraper_config.get('SCRAPER', 'project_name')
    subreddit_list = [e.strip() for e in scraper_config.get(
        'SCRAPER', 'subreddit_list').split(',')]
    include_comments = scraper_config.getboolean('SCRAPER', 'include_comments')
    sorting = scraper_config.get('SCRAPER', 'sorting')
    save_method = scraper_config.get('SCRAPER', 'save_method')
    save_location = scraper_config.get('SCRAPER', 'save_location')
    s3_bucket_name = scraper_config.get('SCRAPER', 's3_bucket_name')
    main()
