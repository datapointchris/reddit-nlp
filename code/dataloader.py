import databases
import pandas as pd
import scraping

def data_selector(subreddit_list, source):
    '''Finds data for subreddits from selected source.
       
    Source: scrape
        Scrape each subreddit in the list now.
        Returns DataFrame
        
    Source: csv
        CSVs for subreddits in the 'scraped_subreddits' directory
        Prints subreddits with no CSV files
        Returns DataFrame
        
    Source: mongo
        Creates connection to Mongo DB
        Queries DB for subreddits
        Returns DataFrame
        
    Source: sqlite3
        Creates connection to SQLite DB
        Queries DB for subreddits
        Returns DataFrame
        
    Source: postgres
        Creates connection to Postgres DB
        Queries DB for subreddits
        Returns DataFrame
        
    Source: mysql
        Creates connection to Mysql DB
        Queries DB for subreddits
        Returns DataFrame

    Returns:
    df - DataFrame of selected data

    MODIFIES:
    'subreddit_list'
    '''

    if source == 'scrape':
        scrape = scraping.Scraper()
        df = scrape.scrape_subreddit(subreddit_list)
        return df

    if source == 'csv':
        df = pd.DataFrame()
        trimmed_list = []
        for sub in subreddit_list:
            csv_files = sorted(glob(f'../scraped_subreddits/*{sub}*.csv'))
            if len(csv_files) > 0:
                trimmed_list.append(sub)
                for csv_file in csv_files:
                    data = pd.read_csv(csv_file)
                    df = pd.concat([df, data], ignore_index=True)
            else:
                print(f'No data for {sub}, not adding to df')
        return df

### NOTE ### this bypasses the 'execute_read_query' function in the databases module...
    if source == 'sqlite':
        db = databases.Sqlite()
        connection = db.create_connection('reddit.sqlite')

        placeholders = ','.join('?' for sub in subreddit_list)

        subreddit_query = f"""
        SELECT 
            title, 
            subreddit, 
            date
        FROM subreddits 
        WHERE subreddit IN (%s);""" % placeholders

        cursor = connection.cursor()
        cursor.execute(subreddit_query, subreddit_list)

        column_names = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data=data, columns=column_names)
        
        for sub in subreddit_list:
            if len(df[df['subreddit'] == sub]) == 0:
                print(f'No data for {sub}, not adding to df')
        return df
    
    if source == 'mongo':
        db = databases.Mongo()
        return db.create_connection()

    if source == 'postgres':
        db = databases.Postgres()
        return db.create_connection()
    
    if source == 'mysql':
        db = databases.Mysql()
        return db.create_connection()


def subreddit_encoder(df):
    '''Encodes each subreddit in the dataframe to numeric as 'sub_code' '''
    topic_dict = {}
    for index, subreddit in enumerate(df.subreddit.unique()):
        topic_dict.update({subreddit: index})
        df['sub_code'] = df['subreddit'].map(topic_dict)
    print(f'Subreddits and codes added: {topic_dict}')
    return df